"""
Use case: answer a user query with hybrid retrieval + local LLM.

Steps:
  query → semantic search + BM25 search → RRF fusion → filter →
  build context → LLM → Answer (optionally with images).
"""
from __future__ import annotations

import logging

from app.application.services.context_builder import build_context
from app.application.services.retrieval_filter import filter_chunks
from app.application.services.rrf_fusion import reciprocal_rank_fusion
from app.config.settings import Settings
from app.domain.entities.answer import Answer, RetrievedContext
from app.domain.entities.chunk import Chunk, ChunkMetadata
from app.domain.entities.image import ExtractedImage
from app.domain.entities.query import Query, QueryIntent
from app.domain.repositories.bm25_index import BM25IndexPort
from app.domain.repositories.embedder import EmbedderPort
from app.domain.repositories.image_store import ImageStorePort
from app.domain.repositories.llm import LLMPort
from app.domain.repositories.vector_store import VectorStorePort
from app.domain.value_objects.ids import ChunkId, DocumentId, ImageId
from app.domain.value_objects.section_path import SectionPath

logger = logging.getLogger("paperlens.application.answer")

_SYSTEM_PROMPT = (
    "You are PaperLens, a research-paper assistant. Answer the user's "
    "question using ONLY the provided context chunks. Cite chunk ids "
    "when useful. If the context does not contain the answer, say so "
    "explicitly instead of guessing."
)


class AnswerQueryUseCase:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
        bm25_index: BM25IndexPort,
        image_store: ImageStorePort,
        llm: LLMPort,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.image_store = image_store
        self.llm = llm

    # ------------------------------------------------------------------ #
    def run(self, query: Query) -> Answer:
        top_k = query.top_k or self.settings.max_chunks * 3

        semantic = self._semantic_search(query.text, top_k)
        lexical = self._bm25_search(query.text, top_k)
        fused = reciprocal_rank_fusion([semantic, lexical], k=self.settings.rrf_k)

        token_counts = {cid: self._token_count(p["text"]) for cid, _, p in fused}
        kept = filter_chunks(
            fused,
            token_counts=token_counts,
            min_relevance_score=self.settings.min_relevance_score,
            max_chunks=self.settings.max_chunks,
            max_context_tokens=self.settings.max_context_tokens,
        )

        chunks = [self._payload_to_chunk(cid, p) for cid, _, p in kept]
        context = RetrievedContext(
            chunks=chunks,
            total_tokens=sum(c.token_count for c in chunks),
        )

        # Resolve images referenced by the kept chunks.
        referenced_images = self._resolve_images(chunks)
        context.images = referenced_images

        # Decide whether to attach images to the final answer.
        attached_images: list[ExtractedImage] = []
        if query.intent in (QueryIntent.IMAGE_ONLY, QueryIntent.TEXT_AND_IMAGE):
            attached_images = referenced_images
        elif query.intent == QueryIntent.TEXT_ONLY and not referenced_images:
            attached_images = []

        context_str = build_context(chunks) if chunks else "(no relevant context found)"
        user_prompt = f"Question:\n{query.text}\n\nContext:\n{context_str}\n\nAnswer:"

        try:
            answer_text = self.llm.generate(_SYSTEM_PROMPT, user_prompt)
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            answer_text = "The local LLM could not generate an answer."

        return Answer(text=answer_text, context=context, used_images=attached_images)

    # ------------------------------------------------------------------ #
    def _semantic_search(self, text: str, top_k: int):
        try:
            qv = self.embedder.embed_query(text)
            return self.vector_store.search(qv, top_k=top_k)
        except Exception as exc:
            logger.warning("Semantic search failed: %s", exc)
            return []

    def _bm25_search(self, text: str, top_k: int):
        try:
            return self.bm25_index.search(text, top_k=top_k)
        except Exception as exc:
            logger.warning("BM25 search failed: %s", exc)
            return []

    def _resolve_images(self, chunks: list[Chunk]) -> list[ExtractedImage]:
        images: list[ExtractedImage] = []
        seen: set[str] = set()
        for c in chunks:
            for image_id in c.metadata.image_ids:
                if image_id in seen:
                    continue
                seen.add(image_id)
                path = self.settings.image_dir / c.document_id / f"{image_id}.png"
                caption = ""
                idx = c.metadata.image_ids.index(image_id)
                if idx < len(c.metadata.image_captions):
                    caption = c.metadata.image_captions[idx]
                images.append(ExtractedImage(
                    image_id=ImageId(image_id),
                    document_id=DocumentId(c.document_id),
                    page_number=c.metadata.page_numbers[0] if c.metadata.page_numbers else 0,
                    section_path=c.metadata.section_path,
                    image_path=str(path),
                    caption=caption,
                ))
        return images

    def _payload_to_chunk(self, chunk_id: str, payload: dict) -> Chunk:
        meta = ChunkMetadata(
            document_id=DocumentId(payload["document_id"]),
            filename=payload.get("filename", ""),
            section_path=SectionPath.of(payload.get("section_path", [])),
            page_numbers=list(payload.get("page_numbers", [])),
            image_ids=[ImageId(x) for x in payload.get("image_ids", [])],
            image_captions=list(payload.get("image_captions", [])),
            table_ids=payload.get("table_ids", []),
            table_captions=list(payload.get("table_captions", [])),
        )
        return Chunk(
            chunk_id=ChunkId(chunk_id),
            document_id=meta.document_id,
            text=payload.get("text", ""),
            metadata=meta,
            token_count=self._token_count(payload.get("text", "")),
        )

    @staticmethod
    def _token_count(text: str) -> int:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return max(1, len(text) // 4)