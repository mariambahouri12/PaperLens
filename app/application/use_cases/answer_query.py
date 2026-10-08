# application/use_cases/answer_query.py
"""
Use case: answer a user query with hybrid retrieval + local LLM.

Steps:
  query -> semantic search + BM25 search -> RRF fusion -> filter ->
  build context -> LLM -> Answer (optionally with images).

Images are referenced by `ChunkMetadata.image_path` (the real path on
disk). They are never copied; the resolver only validates and reads them.
"""
from __future__ import annotations

import logging

from app.application.services.context_builder import build_context
from app.application.services.retrieval_filter import filter_chunks
from app.application.services.rrf_fusion import reciprocal_rank_fusion
from app.config.settings import Settings
from app.domain.entities.answer import Answer, RetrievedContext
from app.domain.entities.chunk import Chunk, ChunkMetadata, ChunkType
from app.domain.entities.image import ExtractedImage
from app.domain.entities.query import Query, QueryIntent
from app.domain.repositories.bm25_index import BM25IndexPort
from app.domain.repositories.embedder import EmbedderPort
from app.domain.repositories.image_resolver import ImageResolverPort
from app.domain.repositories.llm import LLMPort
from app.domain.repositories.vector_store import VectorStorePort
from app.domain.value_objects.ids import ChunkId, DocumentId, ImageId, TableId
from app.domain.value_objects.section_path import SectionPath
from app.infrastructure.chunking.token_counter import count_tokens

logger = logging.getLogger("paperlens.application.answer")

_SYSTEM_PROMPT = (
    "You are PaperLens, a research-paper assistant. "
    "Answer the user's question using ONLY the information contained "
    "in the provided context chunks. "
    "\n\n"
    "Each chunk starts with a header in this exact format: "
    "[document=<filename> | section=<section path> | pages=<page numbers>]. "
    "\n\n"
    "Answering style: "
    "Do not copy or reproduce the context verbatim unless the user explicitly "
    "asks for a quotation. "
    "Instead, understand the relevant information, synthesize it, and "
    "answer the question naturally and concisely in your own words. "
    "Combine information from multiple chunks when necessary to form a "
    "complete answer. "
    "\n\n"
    "Grounding rules: "
    "Use only information supported by the provided context. "
    "Do not use outside knowledge, assumptions, or unstated conclusions. "
    "You may infer a direct conclusion when it follows clearly from the "
    "provided information, but do not extend a claim beyond the conditions "
    "described in the paper. "
    "\n\n"
    "Citation rules: "
    "Cite the supporting source for factual claims using the document name, "
    "section, and page number from the chunk header. "
    "Use only sources that actually appear in the provided headers. "
    "Never invent or modify citation information. "
    "\n\n"
    "Scientific accuracy: "
    "Preserve the meaning and scope of the paper's claims. "
    "Clearly distinguish between proposed methods, experimental results, "
    "limitations, and future work. "
    "\n\n"
    "If the provided context does not contain enough information to answer "
    "the question, say explicitly that the information is not available "
    "in the provided context."
)


class AnswerQueryUseCase:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
        bm25_index: BM25IndexPort,
        image_resolver: ImageResolverPort,
        llm: LLMPort,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.image_resolver = image_resolver
        self.llm = llm

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def run(self, query: Query) -> Answer:
        candidate_k = (
            query.top_k
            or self.settings.max_chunks * self.settings.retrieval_candidate_multiplier
        )

        semantic = self._semantic_search(query.text, candidate_k)
        lexical = self._bm25_search(query.text, candidate_k)
        fused = reciprocal_rank_fusion(
            [semantic, lexical],
            k=self.settings.rrf_k,
        )

        # Token counts are already in the payload (computed once during
        # ingestion), so no need to recompute them here.
        token_counts = {
            chunk_id: int(payload.get("token_count", 0))
            for chunk_id, _, payload in fused
        }

        kept = filter_chunks(
            fused,
            token_counts=token_counts,
            min_relevance_score=self.settings.min_relevance_score,
            max_chunks=self.settings.max_chunks,
            max_context_tokens=self.settings.max_context_tokens,
        )

        chunks = [self._payload_to_chunk(chunk_id, payload) for chunk_id, _, payload in kept]

        context = RetrievedContext(
            chunks=chunks,
            total_tokens=sum(chunk.token_count for chunk in chunks),
        )

        referenced_images = self._resolve_images(chunks)
        context.images = referenced_images

        attached_images = self._select_attached_images(
            query.intent,
            referenced_images,
        )

        context_str = (
            build_context(chunks) if chunks else "(no relevant context found)"
        )
        user_prompt = (
            f"Question:\n{query.text}\n\n"
            f"Context:\n{context_str}\n\nAnswer:"
        )

        try:
            answer_text = self.llm.generate(_SYSTEM_PROMPT, user_prompt)
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            answer_text = "The local LLM could not generate an answer."

        return Answer(
            text=answer_text,
            context=context,
            used_images=attached_images,
        )

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def _semantic_search(self, text: str, top_k: int):
        try:
            query_vector = self.embedder.embed_query(text)
            return self.vector_store.search(query_vector, top_k=top_k)
        except Exception as exc:
            logger.warning("Semantic search failed: %s", exc)
            return []

    def _bm25_search(self, text: str, top_k: int):
        try:
            return self.bm25_index.search(text, top_k=top_k)
        except Exception as exc:
            logger.warning("BM25 search failed: %s", exc)
            return []

    # ------------------------------------------------------------------ #
    # Images
    # ------------------------------------------------------------------ #

    def _resolve_images(self, chunks: list[Chunk]) -> list[ExtractedImage]:
        """
        Collect the images referenced by the kept chunks.

        Deduplication is by `image_path` (the real, stable identifier),
        because `image_id` is derived from the file name and could
        theoretically collide across documents.
        """
        images: list[ExtractedImage] = []
        seen: set[str] = set()

        for chunk in chunks:
            meta = chunk.metadata

            if not meta.image_path or meta.image_path in seen:
                continue

            if not self.image_resolver.exists(meta.image_path):
                logger.warning(
                    "Image referenced by chunk %s is missing: %s",
                    chunk.chunk_id,
                    meta.image_path,
                )
                continue

            seen.add(meta.image_path)

            images.append(
                ExtractedImage(
                    image_id=meta.image_id or ImageId(meta.image_path),
                    document_id=meta.document_id,
                    image_path=meta.image_path,
                    page_number=(
                        meta.page_numbers[0] if meta.page_numbers else 0
                    ),
                    section_path=meta.section_path,
                    caption="",  # caption is inside chunk.text, not separate
                    kind=self._image_kind(meta.chunk_types),
                )
            )

        return images

    @staticmethod
    def _image_kind(chunk_types: tuple[ChunkType, ...]) -> str:
        if ChunkType.CHART in chunk_types:
            return "chart"
        return "image"

    @staticmethod
    def _select_attached_images(
        intent: QueryIntent,
        referenced: list[ExtractedImage],
    ) -> list[ExtractedImage]:
        if intent == QueryIntent.TEXT_ONLY:
            return []
        if intent in (QueryIntent.IMAGE_ONLY, QueryIntent.TEXT_AND_IMAGE):
            return referenced
        return []

    # ------------------------------------------------------------------ #
    # Payload -> Chunk
    # ------------------------------------------------------------------ #

    def _payload_to_chunk(self, chunk_id: str, payload: dict) -> Chunk:
        """
        Rebuild a `Chunk` from a flat payload.

        Only the fields needed downstream are restored; the section path
        is rebuilt as a `SectionPath` value object.
        """
        text = payload.get("text", "")

        meta = ChunkMetadata(
            document_id=DocumentId(payload["document_id"]),
            filename=payload.get("filename", ""),
            chunk_index=int(payload.get("chunk_index", 0)),
            chunk_types=self._parse_types(payload.get("types", "")),
            section_path=SectionPath.of(payload.get("section_path", [])),
            page_numbers=tuple(payload.get("page_numbers", [])),
            image_id=(
                ImageId(payload["image_id"])
                if payload.get("image_id")
                else None
            ),
            image_path=payload.get("image_path"),
            table_id=(
                TableId(payload["table_id"])
                if payload.get("table_id")
                else None
            ),
        )

        return Chunk(
            chunk_id=ChunkId(chunk_id),
            document_id=meta.document_id,
            text=text,
            metadata=meta,
            token_count=int(
                payload.get("token_count") or count_tokens(text)
            ),
        )

    @staticmethod
    def _parse_types(raw) -> tuple[ChunkType, ...]:
        if isinstance(raw, (list, tuple)):
            names = raw
        else:
            names = str(raw or "").split(",")

        parsed: list[ChunkType] = []

        for name in names:
            name = str(name).strip()
            if not name:
                continue
            try:
                parsed.append(ChunkType(name))
            except ValueError:
                logger.warning("Unknown chunk type in payload: %r", name)

        return tuple(parsed)