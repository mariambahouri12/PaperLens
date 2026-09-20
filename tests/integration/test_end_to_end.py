"""
End-to-end smoke test.

Skips automatically if no PDF is present in data/, so it can be run in
any environment without shipping a fixture.
"""
from pathlib import Path

import pytest

from app.application.use_cases.answer_query import AnswerQueryUseCase
from app.application.use_cases.ingest_documents import IngestDocumentsUseCase
from app.config.settings import Settings
from app.domain.entities.query import Query
from app.infrastructure.bm25.rank_bm25_index import RankBM25Index
from app.infrastructure.chunking.hierarchical_chunker import HierarchicalChunker
from app.infrastructure.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from app.infrastructure.extraction.pymupdf_extractor import PyMuPDFExtractor
from app.infrastructure.image_store.filesystem_image_store import FilesystemImageStore
from app.infrastructure.llm.ollama_llm import OllamaLLM
from app.infrastructure.vector_store.qdrant_local_store import QdrantLocalStore


@pytest.mark.skipif(
    not any(Path("data").glob("*.pdf")),
    reason="No PDF in data/ to run the end-to-end test against.",
)
def test_ingest_and_query(tmp_path: Path):
    settings = Settings(
        data_dir=Path("data"),
        image_dir=tmp_path / "images",
        storage_dir=tmp_path / "storage",
    )
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    vector_store = QdrantLocalStore(settings.vector_store_path, dimension=embedder.dimension)
    bm25 = RankBM25Index(settings.bm25_store_path)
    image_store = FilesystemImageStore(settings)

    ingest = IngestDocumentsUseCase(
        settings=settings,
        extractors=[PyMuPDFExtractor()],
        image_store=image_store,
        chunker=HierarchicalChunker(settings),
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25,
    )
    stats = ingest.run()
    assert stats["ingested"] >= 1

    answer_uc = AnswerQueryUseCase(
        settings=settings,
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25,
        image_store=image_store,
        llm=OllamaLLM(settings),
    )
    answer = answer_uc.run(Query(text="What is this paper about?"))
    assert answer.text