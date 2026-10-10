
# interfaces/cli/commands.py
"""
Composition root + CLI for PaperLens.

This is the ONLY module that knows about every concrete adapter.
The CLI commands build the use cases with the required dependencies
and call them.

Nothing in domain/ or application/ imports a specific external
library such as qdrant-client, rank_bm25, sentence-transformers,
ollama, Typer, etc.

Commands
--------
    paperlens ingest [--data-dir PATH] [--force]
    paperlens ask "question" [--top-k N] [--no-images]
    paperlens images "query" [--top-k N]
    paperlens stats
    paperlens --version
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from app.application.use_cases.answer_query import AnswerQueryUseCase
from app.application.use_cases.ingest_documents import (
    IngestDocumentsUseCase,
)
from app.application.use_cases.retrieve_images import (
    RetrieveImagesUseCase,
)
from app.config.settings import Settings, settings as default_settings
from app.domain.entities.answer import Answer
from app.domain.entities.image import ExtractedImage
from app.domain.entities.query import Query, QueryIntent
from app.infrastructure.bm25.rank_bm25_index import RankBM25Index
from app.infrastructure.checkpoint.json_checkpoint import (
    JsonCheckpointRepository,
)
from app.infrastructure.chunking.config import ChunkingConfig
from app.infrastructure.chunking.hierarchical_chunker import (
    HierarchicalChunker,
)
from app.infrastructure.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from app.infrastructure.extraction.extractor import MinerUExtractor
from app.infrastructure.image_store.filesystem_image_resolver import (
    FilesystemImageResolver,
)
from app.infrastructure.llm.ollama_llm import OllamaLLM
from app.infrastructure.logging.structured_logger import (
    configure_logging,
)
from app.infrastructure.vector_store.qdrant_local_store import (
    QdrantLocalStore,
)

logger = logging.getLogger("paperlens.interfaces.cli")

app = typer.Typer(
    name="paperlens",
    help="PaperLens: ask questions about your research papers.",
    no_args_is_help=True,
    add_completion=False,
)

# ----------------------------------------------------------------------
# Version
# ----------------------------------------------------------------------

__version__ = "0.1.0"


def _version_callback(value: bool) -> None:
    """Print the application version and exit."""

    if value:
        typer.echo(f"PaperLens {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            help="Show the version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            "-l",
            help="Logging level (DEBUG, INFO, WARNING, ERROR).",
            envvar="PAPERLENS_LOG_LEVEL",
        ),
    ] = "",
) -> None:
    """
    PaperLens command-line interface.

    Logging is configured once for the whole process.
    Individual commands do not modify logging configuration.
    """

    level = log_level or default_settings.log_level
    configure_logging(level=level)


# ----------------------------------------------------------------------
# Dependency factory
# ----------------------------------------------------------------------


def _build_settings() -> Settings:
    """
    Return the application settings and ensure required directories exist.

    Commands use this function as the single entry point for settings.
    """

    default_settings.ensure_directories()
    return default_settings


def _build_chunker(
    settings: Settings,
) -> HierarchicalChunker:
    """Build the hierarchical document chunker."""

    return HierarchicalChunker(
        ChunkingConfig.from_settings(settings),
    )


def _build_embedder(
    settings: Settings,
) -> SentenceTransformerEmbedder:
    """Build the embedding adapter."""

    return SentenceTransformerEmbedder()


def _build_bm25(
    settings: Settings,
) -> RankBM25Index:
    """Build and load the persistent BM25 index."""

    index = RankBM25Index(
        persist_path=settings.bm25_store_path,
    )
    index.load()
    return index


def _build_reranker(settings: Settings):
    """Build the Cross-Encoder only when reranking is enabled."""

    if not settings.use_reranker:
        return None

    from app.infrastructure.reranking.cross_encoder_reranker import (
        CrossEncoderReranker,
    )

    return CrossEncoderReranker(
        model_name=settings.reranker_model,
        device=settings.reranker_device,
    )


def _build_checkpoint(
    settings: Settings,
) -> JsonCheckpointRepository:
    """Build the persistent checkpoint repository."""

    return JsonCheckpointRepository(settings.checkpoint_path)


def _build_image_resolver(
    settings: Settings,
) -> FilesystemImageResolver:
    """Build the filesystem image resolver."""

    return FilesystemImageResolver(
        allowed_roots=[
            settings.data_dir,
            settings.image_dir,
        ],
    )


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------


@app.command()
def ingest(
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            "-d",
            help=(
                "Directory containing the PDFs to ingest "
                "(default: PAPERLENS_DATA_DIR)."
            ),
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Ignore the checkpoint and re-index every chunk.",
        ),
    ] = False,
) -> None:
    """
    Extract, chunk, embed and index every PDF found
    in the data directory.
    """

    settings = _build_settings()

    if data_dir is not None:
        settings.data_dir = data_dir

    checkpoint = _build_checkpoint(settings)

    if force:
        logger.info("--force set: resetting the checkpoint")
        checkpoint.reset()

    bm25 = _build_bm25(settings)

    with QdrantLocalStore(
        path=settings.vector_store_path,
        dimension=settings.embedding_dimension,
    ) as vector_store:
        use_case = IngestDocumentsUseCase(
            settings=settings,
            extractors=[
                MinerUExtractor(settings=settings),
            ],
            chunker=_build_chunker(settings),
            embedder=_build_embedder(settings),
            vector_store=vector_store,
            bm25_index=bm25,
            checkpoint=checkpoint,
        )

        stats = use_case.run()

    _print_ingest_stats(stats)


@app.command()
def ask(
    question: Annotated[
        str,
        typer.Argument(help="The question to ask."),
    ],
    top_k: Annotated[
        int | None,
        typer.Option(
            "--top-k",
            "-k",
            help="Number of chunks to keep after retrieval (default: MAX_CHUNKS).",
            min=1,
        ),
    ] = None,
    no_images: Annotated[
        bool,
        typer.Option(
            "--no-images",
            help="Never attach images, even if the query mentions a figure.",
        ),
    ] = False,
) -> None:
    """Ask a question using hybrid retrieval, optional reranking and the local LLM."""

    settings = _build_settings()

    intent = (
        QueryIntent.TEXT_ONLY
        if no_images
        else QueryIntent.TEXT_AND_IMAGE
    )

    query = Query(
        text=question,
        intent=intent,
        top_k=top_k,
    )

    bm25 = _build_bm25(settings)

    with QdrantLocalStore(
        path=settings.vector_store_path,
        dimension=settings.embedding_dimension,
    ) as vector_store:
        use_case = AnswerQueryUseCase(
            settings=settings,
            embedder=_build_embedder(settings),
            vector_store=vector_store,
            bm25_index=bm25,
            image_resolver=_build_image_resolver(settings),
            llm=OllamaLLM(settings),
            reranker=_build_reranker(settings),
        )

        answer = use_case.run(query)

    _print_answer(answer)


@app.command()
def images(
    query_text: Annotated[
        str,
        typer.Argument(
            help="The query describing the figure you want.",
        ),
    ],
    top_k: Annotated[
        int | None,
        typer.Option(
            "--top-k",
            "-k",
            help="Maximum number of images to return.",
            min=1,
        ),
    ] = None,
) -> None:
    """
    Return images referenced by the most relevant chunks.

    No LLM call is performed.
    """

    settings = _build_settings()
    bm25 = _build_bm25(settings)

    with QdrantLocalStore(
        path=settings.vector_store_path,
        dimension=settings.embedding_dimension,
    ) as vector_store:
        use_case = RetrieveImagesUseCase(
            settings=settings,
            embedder=_build_embedder(settings),
            vector_store=vector_store,
            bm25_index=bm25,
            image_resolver=_build_image_resolver(settings),
        )

        found = use_case.run(query_text, top_k=top_k)

    _print_images(found)


@app.command()
def stats() -> None:
    """
    Print the size of the vector store, BM25 index
    and checkpoint.
    """

    settings = _build_settings()

    with QdrantLocalStore(
        path=settings.vector_store_path,
        dimension=settings.embedding_dimension,
    ) as vector_store:
        vector_count = vector_store.count()

    bm25 = _build_bm25(settings)
    bm25_count = bm25.count()

    checkpoint = _build_checkpoint(settings)
    checkpoint_count = checkpoint.count()

    typer.echo("PaperLens storage")
    typer.echo(f"  vector store : {vector_count} chunk(s)")
    typer.echo(f"  BM25 index   : {bm25_count} chunk(s)")
    typer.echo(f"  checkpoint   : {checkpoint_count} chunk(s)")
    typer.echo(f"  data dir     : {settings.data_dir}")
    typer.echo(f"  storage dir  : {settings.storage_dir}")


# ----------------------------------------------------------------------
# Pretty printing
# ----------------------------------------------------------------------


def _print_ingest_stats(
    stats: dict[str, int],
) -> None:
    """Print the ingestion summary."""

    typer.echo("")
    typer.echo("Ingestion summary")
    typer.echo(f"  discovered : {stats.get('discovered', 0)}")
    typer.echo(f"  ingested   : {stats.get('ingested', 0)}")
    typer.echo(f"  failed     : {stats.get('failed', 0)}")
    typer.echo(f"  chunks     : {stats.get('chunks', 0)}")


def _print_answer(
    answer: Answer,
) -> None:
    """Print the answer and its attached images."""

    typer.echo("")
    typer.echo("=" * 70)
    typer.echo("ANSWER")
    typer.echo("=" * 70)
    typer.echo(answer.text)

    if not answer.used_images:
        return

    typer.echo("")
    typer.echo("Attached images")

    for image in answer.used_images:
        caption = (image.caption or "").splitlines()[0][:80]

        typer.echo(
            f"  - {image.image_id} "
            f"(p. {image.page_number})"
        )
        typer.echo(f"    {image.image_path}")

        if caption:
            typer.echo(f"    {caption}")


def _print_images(
    images: list[ExtractedImage],
) -> None:
    """Print retrieved images."""

    if not images:
        typer.echo("No image found.")
        return

    typer.echo(f"{len(images)} image(s):")

    for image in images:
        typer.echo(
            f"  - {image.image_id} "
            f"(p. {image.page_number})"
        )
        typer.echo(f"    {image.image_path}")
