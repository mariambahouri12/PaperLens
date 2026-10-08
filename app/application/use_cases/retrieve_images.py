"""
Use case: image-focused retrieval.

Runs the same hybrid retrieval pipeline as AnswerQueryUseCase but
does not call the LLM.

Only retrieved chunks containing an image reference are returned
as ExtractedImage objects.
"""

from __future__ import annotations

import logging
from typing import Any

from app.application.mappers.chunk_payload_mapper import ChunkPayloadMapper
from app.application.services.hybrid_retriever import HybridRetriever
from app.config.settings import Settings
from app.domain.entities.image import ExtractedImage
from app.domain.repositories.bm25_index import BM25IndexPort
from app.domain.repositories.embedder import EmbedderPort
from app.domain.repositories.image_resolver import ImageResolverPort
from app.domain.repositories.vector_store import VectorStorePort
from app.domain.value_objects.ids import DocumentId, ImageId

logger = logging.getLogger(
    "paperlens.application.retrieve_images"
)


class RetrieveImagesUseCase:
    """
    Retrieve images referenced by top-ranked chunks.

    No LLM call is performed.
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

        self.retriever = HybridRetriever(
            embedder=embedder,
            vector_store=vector_store,
            bm25_index=bm25_index,
            rrf_k=settings.rrf_k,
        )

        self.image_resolver = image_resolver
        self.chunk_mapper = ChunkPayloadMapper()

    def run(
        self,
        query_text: str,
        top_k: int | None = None,
    ) -> list[ExtractedImage]:
        if not query_text.strip():
            return []

        candidate_k = (
            top_k
            if top_k is not None
            else (
                self.settings.max_chunks
                * self.settings.retrieval_candidate_multiplier
            )
        )

        fused = self.retriever.retrieve(
            query=query_text,
            top_k=candidate_k,
        )

        limit = (
            top_k
            if top_k is not None
            else self.settings.max_chunks
        )

        if limit <= 0:
            return []

        images: list[ExtractedImage] = []
        seen: set[str] = set()

        for chunk_id, _, payload in fused:
            if len(images) >= limit:
                break

            image_path = payload.get(
                "image_path"
            )

            if not image_path:
                continue

            image_path = str(image_path)

            if image_path in seen:
                continue

            if not self.image_resolver.exists(
                image_path
            ):
                logger.warning(
                    "Image referenced by chunk %s is missing: %s",
                    chunk_id,
                    image_path,
                )
                continue

            image = self._payload_to_image(
                chunk_id,
                payload,
            )

            seen.add(image_path)
            images.append(image)

        logger.info(
            "Image retrieval returned %d image(s) "
            "from %d fused chunk(s)",
            len(images),
            len(fused),
        )

        return images

    def _payload_to_image(
        self,
        chunk_id: str,
        payload: dict[str, Any],
    ) -> ExtractedImage:
        """
        Build an ExtractedImage from a retrieval payload.

        The chunk mapper is intentionally used first so the payload
        validation and domain reconstruction remain centralized.
        """

        chunk = self.chunk_mapper.to_chunk(
            chunk_id,
            payload,
        )

        metadata = chunk.metadata

        image_path = metadata.image_path

        if not image_path:
            raise ValueError(
                f"Chunk '{chunk_id}' does not reference an image"
            )

        return ExtractedImage(
            image_id=(
                metadata.image_id
                or ImageId(image_path)
            ),
            document_id=(
                metadata.document_id
            ),
            image_path=image_path,
            page_number=(
                metadata.page_numbers[0]
                if metadata.page_numbers
                else 0
            ),
            section_path=metadata.section_path,
            caption="",
            kind=(
                "chart"
                if any(
                    chunk_type.value == "chart"
                    for chunk_type in metadata.chunk_types
                )
                else "image"
            ),
        )