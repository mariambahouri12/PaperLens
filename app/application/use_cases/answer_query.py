"""
Use case: answer a user query using hybrid retrieval and a local LLM.

Pipeline:

    query
      ↓
    hybrid retrieval
      ↓
    RRF
      ↓
    filtering
      ↓
    payload → Chunk
      ↓
    context building ([S#] markers)
      ↓
    local LLM
      ↓
    citation formatting (markers → natural verified sources)
      ↓
    image resolution (cited chunks only)
      ↓
    Answer

The LLM only places [S#] markers. Document names, pages and sections are
reconstructed from the verified retrieved chunks, never invented by the LLM.
Only the chunks actually cited are kept in the answer context.
"""

from __future__ import annotations

import logging
from typing import Any

from app.application.mappers.chunk_payload_mapper import ChunkPayloadMapper
from app.application.services.citation_formatter import apply_citations
from app.application.services.context_builder import build_context
from app.application.services.hybrid_retriever import HybridRetriever
from app.application.services.retrieval_filter import filter_chunks
from app.config.settings import Settings
from app.domain.entities.answer import Answer, RetrievedContext
from app.domain.entities.chunk import Chunk, ChunkType
from app.domain.entities.image import ExtractedImage
from app.domain.entities.query import Query, QueryIntent
from app.domain.exceptions import LLMError, RetrievalError
from app.domain.repositories.bm25_index import BM25IndexPort
from app.domain.repositories.embedder import EmbedderPort
from app.domain.repositories.image_resolver import ImageResolverPort
from app.domain.repositories.llm import LLMPort
from app.domain.repositories.vector_store import VectorStorePort
from app.domain.value_objects.ids import ImageId
from app.domain.repositories.reranker import RerankerPort
from app.infrastructure.logging.structured_logger import QueryTrace


logger = logging.getLogger("paperlens.application.answer")


_SYSTEM_PROMPT = (
    "You are PaperLens, a research-paper assistant. "
    "Answer the user's question using ONLY the information contained "
    "in the provided context chunks. "
    "\n\n"
    "Each chunk starts with a header in this exact format: "
    "[S<number> | document=<filename> | section=<section path> | "
    "pages=<page numbers>]. "
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
    "Source citation rules: "
    "Each context chunk starts with a header beginning with a marker such as "
    "[S1], [S2], etc. "
    "After each sentence or claim that relies on a chunk, add that chunk's "
    "marker, for example: 'The model reaches 92% accuracy [S2].' "
    "Cite ONLY the chunks you actually used to write the answer; never cite "
    "a chunk you did not use. "
    "Use only the markers exactly as given. Never write document names, page "
    "numbers or section names yourself, the application will replace the "
    "markers with the precise source. "
    "If the context does not contain the answer, do not add any marker. "
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
    """Application service responsible for answering user queries."""

    def __init__(
        self,
        settings: Settings,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
        bm25_index: BM25IndexPort,
        image_resolver: ImageResolverPort,
        llm: LLMPort,
        reranker: RerankerPort | None = None,
    ) -> None:
        self.settings = settings

        self.retriever = HybridRetriever(
            embedder=embedder,
            vector_store=vector_store,
            bm25_index=bm25_index,
            rrf_k=settings.rrf_k,
            use_bm25=settings.use_bm25,
            use_dense=settings.use_dense,
            use_rrf=settings.use_rrf,
            reranker_candidate_k=settings.reranker_candidate_k,
        )

        self.image_resolver = image_resolver
        self.llm = llm
        self.chunk_mapper = ChunkPayloadMapper()

    def run(
        self,
        query: Query,
    ) -> Answer:
        trace = QueryTrace(logger, query.text)

        candidate_k = self._candidate_k(
            query.top_k
        )

        with trace.stage("retrieval"):
            fused = self.retriever.retrieve(
                query=query.text,
                top_k=candidate_k,
            )

        token_counts = self._token_counts(fused)

        with trace.stage("filtering"):
            kept = filter_chunks(
                fused,
                token_counts=token_counts,
                min_relevance_score=self.settings.min_relevance_score,
                max_chunks=self.settings.max_chunks,
                max_context_tokens=self.settings.max_context_tokens,
            )

        chunks = [
            self.chunk_mapper.to_chunk(
                chunk_id,
                payload,
            )
            for chunk_id, _, payload in kept
        ]

        trace.log_chunks(
            chunks,
            scores=[score for _, score, _ in kept],
            candidates=len(fused),
        )

        context_str = (
            build_context(chunks)
            if chunks
            else "(no relevant context found)"
        )

        user_prompt = (
            f"Question:\n{query.text}\n\n"
            f"Context:\n{context_str}\n\n"
            "Answer:"
        )

        with trace.stage("llm", resources=True):
            raw_answer = self._generate_answer(
                user_prompt
            )

        trace.log_llm(getattr(self.llm, "last_stats", None))

        # Replace [S#] markers by natural verified sources and keep
        # only the chunks that were really cited.
        with trace.stage("citations"):
            answer_text, cited_chunks = apply_citations(
                raw_answer,
                chunks,
            )

            context = RetrievedContext(
                chunks=cited_chunks,
                total_tokens=sum(
                    chunk.token_count
                    for chunk in cited_chunks
                ),
            )

            referenced_images = self._resolve_images(
                cited_chunks
            )
            context.images = referenced_images

            attached_images = self._select_attached_images(
                query.intent,
                referenced_images,
            )

        trace.log_citations(
            raw_answer,
            answer_text,
            chunks,
            cited_chunks,
        )
        trace.finish()

        return Answer(
            text=answer_text,
            context=context,
            used_images=attached_images,
        )

    def _candidate_k(
        self,
        requested_top_k: int | None,
    ) -> int:
        if requested_top_k is not None:
            return requested_top_k

        return (
            self.settings.max_chunks
            * self.settings.retrieval_candidate_multiplier
        )

    @staticmethod
    def _token_counts(
        fused: list[tuple[str, float, dict[str, Any]]],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}

        for chunk_id, _, payload in fused:
            if "token_count" not in payload:
                raise RetrievalError(
                    f"Missing token_count for chunk '{chunk_id}'"
                )

            try:
                token_count = int(
                    payload["token_count"]
                )
            except (TypeError, ValueError) as exc:
                raise RetrievalError(
                    f"Invalid token_count for chunk '{chunk_id}'"
                ) from exc

            if token_count < 0:
                raise RetrievalError(
                    f"Negative token_count for chunk '{chunk_id}'"
                )

            counts[chunk_id] = token_count

        return counts

    def _generate_answer(
        self,
        user_prompt: str,
    ) -> str:
        try:
            return self.llm.generate(
                _SYSTEM_PROMPT,
                user_prompt,
            )
        except LLMError as exc:
            logger.error(
                "LLM generation failed: %s",
                exc,
            )
            return (
                "The local LLM could not generate an answer."
            )

    def _resolve_images(
        self,
        chunks: list[Chunk],
    ) -> list[ExtractedImage]:
        """
        Resolve unique images referenced by the given chunks.

        Deduplication uses image_path because it is the actual stable
        filesystem identifier.
        """

        images: list[ExtractedImage] = []
        seen: set[str] = set()

        for chunk in chunks:
            metadata = chunk.metadata
            image_path = metadata.image_path

            if not image_path:
                continue

            if image_path in seen:
                continue

            if not self.image_resolver.exists(
                image_path
            ):
                logger.warning(
                    "Image referenced by chunk %s is missing: %s",
                    chunk.chunk_id,
                    image_path,
                )
                continue

            seen.add(image_path)

            images.append(
                ExtractedImage(
                    image_id=(
                        metadata.image_id
                        or ImageId(image_path)
                    ),
                    document_id=metadata.document_id,
                    image_path=image_path,
                    page_number=(
                        metadata.page_numbers[0]
                        if metadata.page_numbers
                        else 0
                    ),
                    section_path=metadata.section_path,
                    caption="",
                    kind=self._image_kind(
                        metadata.chunk_types
                    ),
                )
            )

        return images

    @staticmethod
    def _image_kind(
        chunk_types: tuple[ChunkType, ...],
    ) -> str:
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

        if intent in (
            QueryIntent.IMAGE_ONLY,
            QueryIntent.TEXT_AND_IMAGE,
        ):
            return referenced

        return []