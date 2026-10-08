# application/use_cases/ingest_documents.py
"""
Use case: ingest every supported document in the input directory.

Pipeline:
    discover -> extract -> chunk -> dump chunks -> embed ->
    index (vector + BM25) -> checkpoint

Output layout:
    data/output/<stem>_chunking.json    ← ONLY the chunking result
    data/debug/<stem>.json              ← extraction JSON
    data/mineru_raw/<stem>/...          ← raw MinerU (images included)

The text sent to the DENSE encoder is prefixed with the chunk's section
path (e.g. `[section: II.A FedAvg]`) so the semantic search can
distinguish chunks that use similar wording but live in different
sections. BM25 and the payload keep the raw `chunk.text`.
"""
from __future__ import annotations

import logging
import os

from app.config.settings import Settings
from app.domain.entities.chunk import Chunk
from app.domain.entities.document import Document
from app.domain.exceptions import ExtractionError, IndexingError
from app.domain.repositories.bm25_index import BM25IndexPort
from app.domain.repositories.checkpoint import CheckpointRepository
from app.domain.repositories.chunker import ChunkerPort
from app.domain.repositories.document_extractor import DocumentExtractorPort
from app.domain.repositories.embedder import EmbedderPort
from app.domain.repositories.vector_store import VectorStorePort
from app.infrastructure.chunking.chunk_serializer import write_chunks

logger = logging.getLogger("paperlens.application.ingest")


class IngestDocumentsUseCase:
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

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

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
        }

        for path in pdfs:
            try:
                document = self._extract(path)
                chunks = self.chunker.chunk(document)

                self._dump_chunks(document, chunks)
                self._index_chunks(chunks)

                stats["ingested"] += 1
                stats["chunks"] += len(chunks)

                logger.info(
                    "Ingested %s: %d chunk(s)",
                    document.metadata.document_id,
                    len(chunks),
                )

            except (ExtractionError, IndexingError) as exc:
                stats["failed"] += 1
                logger.error("Failed to ingest %s: %s", path, exc)

            except Exception as exc:
                stats["failed"] += 1
                logger.exception(
                    "Unexpected error ingesting %s: %s",
                    path,
                    exc,
                )

        try:
            self.bm25_index.persist()
        except IndexingError as exc:
            logger.error("Could not persist BM25 index: %s", exc)

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

        raise ExtractionError(f"No extractor supports '{path}'")

    # ------------------------------------------------------------------ #
    # Chunk dump
    # ------------------------------------------------------------------ #

    def _dump_chunks(self, document: Document, chunks: list[Chunk]) -> None:
        """
        Write the chunking result to
        `<data_dir>/output/<stem>_chunking.json`.
        """
        output_dir = self.settings.output_dir
        stem = self._stem_for(document)
        target = output_dir / f"{stem}_chunking.json"

        try:
            write_chunks(chunks, target)
            logger.info("Chunks written: %s", target)
        except Exception as exc:
            logger.warning("Could not write chunk dump %s: %s", target, exc)

    @staticmethod
    def _stem_for(document: Document) -> str:
        """Return the source PDF stem from `document.metadata.filename`."""
        filename = document.metadata.filename
        if filename:
            return filename.rsplit(".", 1)[0]
        return document.metadata.document_id

    # ------------------------------------------------------------------ #
    # Chunk indexing
    # ------------------------------------------------------------------ #

    def _index_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        batch_size = self.settings.embedding_batch_size

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]

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

            ids = [chunk.chunk_id for chunk in pending]

            # --- Text used for the DENSE encoder: section path + text.
            embedding_texts = [
                self._embedding_text(chunk) for chunk in pending
            ]

            # --- Text used for BM25 and stored in the payload: raw text.
            bm25_texts = [chunk.text for chunk in pending]

            payloads = [self._chunk_to_payload(chunk) for chunk in pending]

            logger.info(
                "Embedding batch %d-%d (%d chunk(s)).",
                start,
                start + len(batch) - 1,
                len(pending),
            )

            vectors = self.embedder.embed_documents(embedding_texts)
            self.vector_store.upsert(ids, vectors, payloads)
            self.bm25_index.add(ids, bm25_texts, payloads)
            self.checkpoint.mark_completed(ids)

            logger.info(
                "Completed batch %d-%d (%d chunk(s)).",
                start,
                start + len(batch) - 1,
                len(pending),
            )

    @staticmethod
    def _embedding_text(chunk: Chunk) -> str:
        """
        Build the text that will be embedded by the dense encoder.

        The section path is prepended as a short context header so the
        semantic search can distinguish chunks that live in different
        sections but use similar wording.

        The original `chunk.text` is NOT modified: it stays what the
        LLM sees and what is stored in the payload.
        """
        section = chunk.metadata.section_path.as_string()
        if not section:
            return chunk.text

        return f"[section: {section}]\n\n{chunk.text}"

    # ------------------------------------------------------------------ #
    # Payload construction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _chunk_to_payload(chunk: Chunk) -> dict:
        meta = chunk.metadata

        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "text": chunk.text,
            "token_count": chunk.token_count,
            "filename": meta.filename,
            "chunk_index": meta.chunk_index,
            "types": ",".join(t.value for t in meta.chunk_types),
            "section_path": meta.section_path.as_list(),
            "page_numbers": list(meta.page_numbers),
            "image_id": meta.image_id,
            "image_path": meta.image_path,
        }