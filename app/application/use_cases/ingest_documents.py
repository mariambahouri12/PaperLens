"""
Use case: ingest every supported document in the input directory.

Pipeline:
discover → extract → persist images → chunk → embed → index (vector + BM25)
                                                        ↓
                                                   checkpoint

Embedding is performed in batches so that completed batches can be
resumed after an interruption.
"""
from __future__ import annotations

import logging
import os

from app.config.settings import Settings
from app.domain.entities.document import Document
from app.domain.exceptions import ExtractionError, IndexingError
from app.domain.repositories.bm25_index import BM25IndexPort
from app.domain.repositories.checkpoint import CheckpointRepository
from app.domain.repositories.chunker import ChunkerPort
from app.domain.repositories.document_extractor import DocumentExtractorPort
from app.domain.repositories.embedder import EmbedderPort
from app.domain.repositories.image_store import ImageStorePort
from app.domain.repositories.vector_store import VectorStorePort

logger = logging.getLogger("paperlens.application.ingest")


class IngestDocumentsUseCase:
    def __init__(
        self,
        settings: Settings,
        extractors: list[DocumentExtractorPort],
        image_store: ImageStorePort,
        chunker: ChunkerPort,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
        bm25_index: BM25IndexPort,
        checkpoint: CheckpointRepository,
    ) -> None:
        self.settings = settings
        self.extractors = extractors
        self.image_store = image_store
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.checkpoint = checkpoint

    def run(self) -> dict:
        self.settings.ensure_directories()

        pdfs = self._discover_pdfs()

        logger.info(
            "Discovered %d document(s) in %s",
            len(pdfs),
            self.settings.data_dir,
        )

        stats = {
            "discovered": len(pdfs),
            "ingested": 0,
            "failed": 0,
            "chunks": 0,
            "images": 0,
        }

        for path in pdfs:
            try:
                doc = self._extract(path)

                self._persist_images(doc)

                chunks = self.chunker.chunk(doc)

                self._index_chunks(chunks)

                stats["ingested"] += 1
                stats["chunks"] += len(chunks)
                stats["images"] += len(doc.images)

                logger.info(
                    "Ingested %s: %d chunk(s), %d image(s)",
                    doc.metadata.document_id,
                    len(chunks),
                    len(doc.images),
                )

            except (ExtractionError, IndexingError) as exc:
                stats["failed"] += 1

                logger.error(
                    "Failed to ingest %s: %s",
                    path,
                    exc,
                )

            except Exception as exc:
                # Never kill the entire ingestion batch because
                # one document failed.
                stats["failed"] += 1

                logger.exception(
                    "Unexpected error ingesting %s: %s",
                    path,
                    exc,
                )

        return stats

    # ------------------------------------------------------------------ #
    # Document discovery
    # ------------------------------------------------------------------ #

    def _discover_pdfs(self) -> list[str]:
        found: list[str] = []

        for root, _, files in os.walk(self.settings.data_dir):
            for name in sorted(files):
                if name.lower().endswith(".pdf"):
                    found.append(os.path.join(root, name))

        return found

    # ------------------------------------------------------------------ #
    # Extraction
    # ------------------------------------------------------------------ #

    def _extract(self, path: str) -> Document:
        for extractor in self.extractors:
            if extractor.supports(path):
                return extractor.extract(path)

        raise ExtractionError(
            f"No extractor supports '{path}'"
        )

    # ------------------------------------------------------------------ #
    # Image persistence
    # ------------------------------------------------------------------ #

    def _persist_images(self, doc: Document) -> None:
        for image in doc.images:
            if self.image_store.exists(image.image_id):
                continue

            raw = image.extra.pop("_raw_bytes", None)

            if raw is None:
                logger.warning(
                    "Image %s has no raw bytes; skipping persist.",
                    image.image_id,
                )
                continue

            try:
                self.image_store.save(image, raw)

            except Exception as exc:
                logger.warning(
                    "Could not persist image %s: %s",
                    image.image_id,
                    exc,
                )

    # ------------------------------------------------------------------ #
    # Chunk indexing
    # ------------------------------------------------------------------ #

    def _index_chunks(self, chunks) -> None:
        if not chunks:
            return

        batch_size = self.settings.embedding_batch_size

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]

            # ---------------------------------------------------------- #
            # Resume support
            # ---------------------------------------------------------- #
            # Skip chunks that were already successfully indexed in a
            # previous run.
            pending = [
                chunk
                for chunk in batch
                if not self.checkpoint.is_completed(chunk.chunk_id)
            ]

            if not pending:
                logger.info(
                    "Skipping completed batch %d-%d.",
                    start,
                    start + len(batch) - 1,
                )
                continue

            ids = [
                chunk.chunk_id
                for chunk in pending
            ]

            texts = [
                chunk.text
                for chunk in pending
            ]

            payloads = [
                self._chunk_to_payload(chunk)
                for chunk in pending
            ]

            logger.info(
                "Embedding batch %d-%d (%d chunk(s)).",
                start,
                start + len(batch) - 1,
                len(pending),
            )

            # ---------------------------------------------------------- #
            # 1. Embed
            # ---------------------------------------------------------- #

            vectors = self.embedder.embed_documents(texts)

            # ---------------------------------------------------------- #
            # 2. Persist vector index
            # ---------------------------------------------------------- #

            self.vector_store.upsert(
                ids,
                vectors,
                payloads,
            )

            # ---------------------------------------------------------- #
            # 3. Persist BM25 index
            # ---------------------------------------------------------- #

            self.bm25_index.add(
                ids,
                texts,
                payloads,
            )

            # ---------------------------------------------------------- #
            # 4. Mark the batch as completed
            # ---------------------------------------------------------- #
            # This MUST happen only after vector + BM25 indexing
            # succeeded.
            self.checkpoint.mark_completed(ids)

            logger.info(
                "Completed embedding/indexing batch %d-%d (%d chunk(s)).",
                start,
                start + len(batch) - 1,
                len(pending),
            )

    # ------------------------------------------------------------------ #
    # Payload construction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _chunk_to_payload(chunk) -> dict:
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "text": chunk.text,
            "filename": chunk.metadata.filename,
            "section_path": chunk.metadata.section_path.as_list(),
            "page_numbers": chunk.metadata.page_numbers,
            "image_ids": list(chunk.metadata.image_ids),
            "image_captions": list(chunk.metadata.image_captions),
            "table_ids": list(chunk.metadata.table_ids),
            "table_captions": list(chunk.metadata.table_captions),
        }