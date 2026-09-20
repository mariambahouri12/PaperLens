"""
CLI interface for PaperLens.

Commands:
  ingest           – build indices from data/*.pdf
  ask <question>   – answer a question (optionally with images)
  show-image <id>  – open an image from the store
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys

import typer
from rich.console import Console

from app.application.use_cases.answer_query import AnswerQueryUseCase
from app.application.use_cases.ingest_documents import IngestDocumentsUseCase
from app.application.use_cases.retrieve_images import RetrieveImagesUseCase
from app.config.settings import settings
from app.domain.entities.query import Query, QueryIntent
from app.infrastructure.bm25.rank_bm25_index import RankBM25Index
from app.infrastructure.chunking.hierarchical_chunker import HierarchicalChunker
from app.infrastructure.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from app.infrastructure.extraction.pymupdf_extractor import PyMuPDFExtractor
from app.infrastructure.image_store.filesystem_image_store import FilesystemImageStore
from app.infrastructure.llm.ollama_llm import OllamaLLM
from app.infrastructure.logging.structured_logger import configure_logging
from app.infrastructure.vector_store.qdrant_local_store import QdrantLocalStore

logger = logging.getLogger("paperlens.interfaces.cli")
console = Console()
app = typer.Typer(help="PaperLens — multimodal RAG for research papers.")


def _wire():
    configure_logging(settings.log_level)
    settings.ensure_directories()

    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    vector_store = QdrantLocalStore(settings.vector_store_path, dimension=embedder.dimension)
    bm25 = RankBM25Index(settings.bm25_store_path)
    bm25.load()

    image_store = FilesystemImageStore(settings)
    llm = OllamaLLM(settings)

    chunker = HierarchicalChunker(settings)
    extractor = PyMuPDFExtractor()

    ingest_uc = IngestDocumentsUseCase(
        settings=settings,
        extractors=[extractor],
        image_store=image_store,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25,
    )
    answer_uc = AnswerQueryUseCase(
        settings=settings,
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25,
        image_store=image_store,
        llm=llm,
    )
    return ingest_uc, answer_uc, bm25, image_store


# ---------------------------------------------------------------------- #
@app.command()
def ingest() -> None:
    """Ingest every PDF under data/ and build the indices."""
    ingest_uc, _, bm25, _ = _wire()
    stats = ingest_uc.run()
    bm25.persist()
    console.print_json(json.dumps(stats))


@app.command()
def ask(
    question: str = typer.Argument(...),
    with_images: bool = typer.Option(False, "--with-images", help="Attach referenced images."),
    image_only: bool = typer.Option(False, "--image-only", help="Return images only."),
) -> None:
    """Ask a question about the ingested papers."""
    _, answer_uc, _, image_store = _wire()

    if image_only:
        retrieve_images = RetrieveImagesUseCase(answer_uc)
        images = retrieve_images.run(question)
        for img in images:
            console.print(f"[bold]{img.image_id}[/bold]  {img.caption}  -> {image_store.resolve_path(img)}")
        return

    intent = QueryIntent.TEXT_AND_IMAGE if with_images else QueryIntent.TEXT_ONLY
    answer = answer_uc.run(Query(text=question, intent=intent))
    console.print("\n[bold green]Answer:[/bold green]\n")
    console.print(answer.text)
    if answer.used_images:
        console.print("\n[bold]Referenced images:[/bold]")
        for img in answer.used_images:
            console.print(f"  - {img.image_id}: {img.caption or '(no caption)'}")
            console.print(f"    {image_store.resolve_path(img)}")


@app.command("show-image")
def show_image(image_id: str) -> None:
    """Open an image from the store using the OS default viewer."""
    from app.domain.entities.image import ExtractedImage
    from app.domain.value_objects.ids import DocumentId, ImageId
    from app.domain.value_objects.section_path import SectionPath

    _, _, _, image_store = _wire()
    placeholder = ExtractedImage(
        image_id=ImageId(image_id),
        document_id=DocumentId(""),
        page_number=0,
        section_path=SectionPath.empty(),
        image_path="",
    )
    path = image_store.resolve_path(placeholder)
    if not path or not sys.platform:
        console.print(f"[red]Image {image_id} not found.[/red]")
        return
    console.print(f"Opening {path}")
    if sys.platform.startswith("win"):
        subprocess.Popen(["start", "", path], shell=True)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])