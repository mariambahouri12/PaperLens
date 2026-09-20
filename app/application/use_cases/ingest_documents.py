"""
Use case: ingest every supported document in the input directory.

Pipeline: discover → extract → chunk → embed → index (vector + BM25).
Image assets are saved to the image store during extraction.
"""
from __future__ import annotations

import logging
import os

from app.config.settings import Settings
from app.domain.entities.document import Document
from app.domain.exceptions import ExtractionError, IndexingError
from app.domain.repositories.bm25_index import BM25IndexPort
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
    ) -> None:
        self.settings = settings
        self.extractors = extractors
        self.image_store = image_store
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def run(self) -> dict:
        self.settings.ensure_directories()
        pdfs = self._discover_pdfs()
        logger.info("Discovered %d document(s) in %s", len(pdfs), self.settings.data_dir)

        stats = {"discovered": len(pdfs), "ingested": 0, "failed": 0, "chunks": 0, "images": 0}

        for path in pdfs:
            try:
                doc = self._extract(path)
                self._persist_images(doc)
                chunks = self.chunker.chunk(doc)
                self._index_chunks(chunks)
                stats["ingested"] += 1
                stats["chunks"] += len(chunks)
                stats["images"] += len(doc.images)
                logger.info("Ingested %s: %d chunk(s), %d image(s)",
                            doc.metadata.document_id, len(chunks), len(doc.images))
            except (ExtractionError, IndexingError) as exc:
                stats["failed"] += 1
                logger.error("Failed to ingest %s: %s", path, exc)
            except Exception as exc:  # never kill the batch
                stats["failed"] += 1
                logger.exception("Unexpected error ingesting %s: %s", path, exc)

        return stats

    # ------------------------------------------------------------------ #
    def _discover_pdfs(self) -> list[str]:
        found: list[str] = []
        for root, _, files in os.walk(self.settings.data_dir):
            for name in sorted(files):
                if name.lower().endswith(".pdf"):
                    found.append(os.path.join(root, name))
        return found

    def _extract(self, path: str) -> Document:
        for extractor in self.extractors:
            if extractor.supports(path):
                return extractor.extract(path)
        raise ExtractionError(f"No extractor supports '{path}'")

    def _persist_images(self, doc: Document) -> None:
        for image in doc.images:
            if self.image_store.exists(image.image_id):
                continue
            raw = image.extra.pop("_raw_bytes", None)
            if raw is None:
                logger.warning("Image %s has no raw bytes; skipping persist.", image.image_id)
                continue
            try:
                self.image_store.save(image, raw)
            except Exception as exc:
                logger.warning("Could not persist image %s: %s", image.image_id, exc)

    def _index_chunks(self, chunks) -> None:
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        payloads = [self._chunk_to_payload(c) for c in chunks]

        vectors = self.embedder.embed_documents(texts)
        self.vector_store.upsert(ids, vectors, payloads)
        self.bm25_index.add(ids, texts, payloads)

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