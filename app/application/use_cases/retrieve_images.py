# application/use_cases/retrieve_images.py
"""
Use case: image-focused retrieval.

Given a query that explicitly asks for a figure/diagram, run the SAME
hybrid retrieval as `AnswerQueryUseCase` and return only the images
referenced by the top chunks. No LLM call is made.

This keeps image retrieval aligned with the text pipeline (same index,
same embeddings) without needing a multimodal vector DB.
"""
from __future__ import annotations

import logging

from app.application.services.rrf_fusion import reciprocal_rank_fusion
from app.config.settings import Settings
from app.domain.entities.image import ExtractedImage
from app.domain.repositories.bm25_index import BM25IndexPort
from app.domain.repositories.embedder import EmbedderPort
from app.domain.repositories.image_resolver import ImageResolverPort
from app.domain.repositories.vector_store import VectorStorePort
from app.domain.value_objects.ids import DocumentId, ImageId

logger = logging.getLogger("paperlens.application.retrieve_images")


class RetrieveImagesUseCase:
    """
    Retrieve images referenced by the top-ranked chunks.

    Differs from `AnswerQueryUseCase` in two ways:
      - it never calls the LLM,
      - it returns only the images, not a textual answer.
    """

    def __init__(
        self,
        settings: Settings,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
        bm25_index: BM25IndexPort,
        image_resolver: ImageResolverPort,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.image_resolver = image_resolver

    def run(self, query_text: str, top_k: int | None = None) -> list[ExtractedImage]:
        candidate_k = (
            top_k
            or self.settings.max_chunks * self.settings.retrieval_candidate_multiplier
        )

        semantic = self._semantic_search(query_text, candidate_k)
        lexical = self._bm25_search(query_text, candidate_k)
        fused = reciprocal_rank_fusion(
            [semantic, lexical],
            k=self.settings.rrf_k,
        )

        # Only keep chunks that carry an image_path.
        images: list[ExtractedImage] = []
        seen: set[str] = set()
        limit = top_k or self.settings.max_chunks

        for chunk_id, _, payload in fused:
            if len(images) >= limit:
                break

            image_path = payload.get("image_path")

            if not image_path or image_path in seen:
                continue

            if not self.image_resolver.exists(image_path):
                logger.warning(
                    "Image referenced by chunk %s is missing: %s",
                    chunk_id,
                    image_path,
                )
                continue

            seen.add(image_path)

            page_numbers = payload.get("page_numbers") or []
            section_path = payload.get("section_path") or []
            types = str(payload.get("types") or "")

            images.append(
                ExtractedImage(
                    image_id=ImageId(
                        payload.get("image_id") or image_path
                    ),
                    document_id=DocumentId(payload["document_id"]),
                    image_path=image_path,
                    page_number=page_numbers[0] if page_numbers else 0,
                    section_path=__import__(
                        "app.domain.value_objects.section_path",
                        fromlist=["SectionPath"],
                    ).SectionPath.of(section_path),
                    caption=payload.get("text", ""),
                    kind="chart" if "chart" in types else "image",
                )
            )

        logger.info(
            "Image query returned %d image(s) out of %d fused chunk(s)",
            len(images),
            len(fused),
        )

        return images

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