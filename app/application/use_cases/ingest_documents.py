"""
Use case: ingest all supported documents from the input directory.

Pipeline:

    discover
      ↓
    extract
      ↓
    chunk
      ↓
    optional chunk dump
      ↓
    embed
      ↓
    vector index
      ↓
    BM25 index
      ↓
    BM25 persistence
      ↓
    checkpoint

The checkpoint is updated only after the required indexes have been
successfully persisted.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.config.settings import Settings
from app.domain.entities.chunk import Chunk
from app.domain.entities.document import Document
from app.domain.exceptions import (
    ExtractionError,
    IndexingError,
)
from app.domain.repositories.bm25_index import BM25IndexPort
from app.domain.repositories.checkpoint import CheckpointRepository
from app.domain.repositories.chunker import ChunkerPort
from app.domain.repositories.document_extractor import (
    DocumentExtractorPort,
)
from app.domain.repositories.embedder import EmbedderPort
from app.domain.repositories.vector_store import VectorStorePort
from app.infrastructure.chunking.chunk_serializer import write_chunks

logger = logging.getLogger(
    "paperlens.application.ingest"
)


class IngestDocumentsUseCase:
    """Orchestrate the complete document ingestion pipeline."""

    def __init__(
        self,
        settings: Settings,
        extractors: list[DocumentExtractorPort],
        chunker: ChunkerPort,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
        bm25_index: BM25IndexPort,
        checkpoint: CheckpointRepository,
    ) -> None:
        self.settings = settings
        self.extractors = extractors
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.checkpoint = checkpoint

    def run(self) -> dict[str, int]:
        self.settings.ensure_directories()

        documents = self._discover_pdfs()

        logger.info(
            "Discovered %d document(s) in %s",
            len(documents),
            self.settings.data_dir,
        )

        stats = {
            "discovered": len(documents),
            "ingested": 0,
            "failed": 0,
            "chunks": 0,
        }

        for path in documents:
            try:
                document = self._extract(path)
                chunks = self.chunker.chunk(document)

                self._dump_chunks(
                    document,
                    chunks,
                )

                self._index_chunks(chunks)

                stats["ingested"] += 1
                stats["chunks"] += len(chunks)

                logger.info(
                    "Ingested %s: %d chunk(s)",
                    document.metadata.document_id,
                    len(chunks),
                )

            except (
                ExtractionError,
                IndexingError,
            ) as exc:
                stats["failed"] += 1

                logger.error(
                    "Failed to ingest %s: %s",
                    path,
                    exc,
                )

            except Exception:
                stats["failed"] += 1

                logger.exception(
                    "Unexpected error ingesting %s",
                    path,
                )

        return stats

    def _discover_pdfs(self) -> list[Path]:
        """Recursively discover PDF files."""

        found: list[Path] = []

        for path in self.settings.data_dir.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() == ".pdf"
            ):
                found.append(path)

        found.sort()

        return found

    def _extract(
        self,
        path: Path,
    ) -> Document:
        """Select the first extractor supporting the file."""

        path_string = str(path)

        for extractor in self.extractors:
            if extractor.supports(path_string):
                return extractor.extract(
                    path_string
                )

        raise ExtractionError(
            f"No extractor supports '{path}'"
        )

    def _dump_chunks(
        self,
        document: Document,
        chunks: list[Chunk],
    ) -> None:
        """
        Write a debug chunk dump.

        This artifact is intentionally non-critical:
        ingestion continues if writing the debug file fails.
        """

        output_dir = self.settings.output_dir

        stem = self._stem_for(document)

        target = (
            output_dir
            / f"{stem}_chunking.json"
        )

        try:
            write_chunks(
                chunks,
                target,
            )

            logger.info(
                "Chunks written: %s",
                target,
            )

        except Exception:
            logger.exception(
                "Could not write chunk dump %s",
                target,
            )

    @staticmethod
    def _stem_for(
        document: Document,
    ) -> str:
        filename = document.metadata.filename

        if filename:
            return filename.rsplit(
                ".",
                1,
            )[0]

        return str(
            document.metadata.document_id
        )

    def _index_chunks(
        self,
        chunks: list[Chunk],
    ) -> None:
        if not chunks:
            return

        batch_size = (
            self.settings.embedding_batch_size
        )

        if batch_size <= 0:
            raise IndexingError(
                "embedding_batch_size must be greater than zero"
            )

        for start in range(
            0,
            len(chunks),
            batch_size,
        ):
            batch = chunks[
                start : start + batch_size
            ]

            pending = [
                chunk
                for chunk in batch
                if not self.checkpoint.is_completed(
                    str(chunk.chunk_id)
                )
            ]

            if not pending:
                logger.info(
                    "Skipping completed batch %d-%d.",
                    start,
                    start + len(batch) - 1,
                )
                continue

            ids = [
                str(chunk.chunk_id)
                for chunk in pending
            ]

            embedding_texts = [
                self._embedding_text(chunk)
                for chunk in pending
            ]

            bm25_texts = [
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

            vectors = (
                self.embedder.embed_documents(
                    embedding_texts
                )
            )

            if len(vectors) != len(pending):
                raise IndexingError(
                    "Embedding count does not match "
                    "pending chunk count"
                )

            # 1. Persist semantic index.
            self.vector_store.upsert(
                ids,
                vectors,
                payloads,
            )

            # 2. Add lexical index.
            self.bm25_index.add(
                ids,
                bm25_texts,
                payloads,
            )

            # 3. Persist lexical index BEFORE checkpoint.
            self.bm25_index.persist()

            # 4. Only now mark the chunks completed.
            self.checkpoint.mark_completed(
                ids
            )

            logger.info(
                "Completed batch %d-%d (%d chunk(s)).",
                start,
                start + len(batch) - 1,
                len(pending),
            )

    @staticmethod
    def _embedding_text(
        chunk: Chunk,
    ) -> str:
        """
        Build the text sent to the dense encoder.

        The original chunk text remains unchanged.
        """

        section = (
            chunk.metadata.section_path.as_string()
        )

        if not section:
            return chunk.text

        return (
            f"[section: {section}]\n\n"
            f"{chunk.text}"
        )

    @staticmethod
    def _chunk_to_payload(
        chunk: Chunk,
    ) -> dict:
        """Serialize a Chunk into an index payload."""

        metadata = chunk.metadata

        return {
            "chunk_id": str(chunk.chunk_id),
            "document_id": str(chunk.document_id),
            "title": metadata.title, 
            "text": chunk.text,
            "token_count": chunk.token_count,
            "filename": metadata.filename,
            "chunk_index": metadata.chunk_index,
            "types": ",".join(
                chunk_type.value
                for chunk_type in metadata.chunk_types
            ),
            "section_path": (
                metadata.section_path.as_list()
            ),
            "page_numbers": list(
                metadata.page_numbers
            ),
            "image_id": (
                str(metadata.image_id)
                if metadata.image_id
                else None
            ),
            "image_path": metadata.image_path,
            "table_id": (
                str(metadata.table_id)
                if metadata.table_id
                else None
            ),
        }
    