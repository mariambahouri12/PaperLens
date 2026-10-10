# streamlit_app.py
"""
PaperLens Streamlit UI.

Run:
    streamlit run streamlit_app.py

The Streamlit layer is responsible only for:
    - UI rendering
    - session state
    - composition of application dependencies

Business and retrieval logic remain in the application/domain layers.
Sources are written naturally inside the answer text by the application
layer, so the UI does not render a separate source block.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import streamlit as st

from app.application.use_cases.answer_query import AnswerQueryUseCase
from app.config.settings import settings
from app.domain.entities.answer import Answer
from app.domain.entities.query import Query, QueryIntent
from app.infrastructure.bm25.rank_bm25_index import RankBM25Index
from app.infrastructure.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from app.infrastructure.image_store.filesystem_image_resolver import (
    FilesystemImageResolver,
)
from app.infrastructure.llm.ollama_llm import OllamaLLM
from app.infrastructure.logging.structured_logger import configure_logging
from app.infrastructure.vector_store.qdrant_local_store import (
    QdrantLocalStore,
)

logger = logging.getLogger("paperlens.interfaces.streamlit")

st.set_page_config(
    page_title="PaperLens",
    page_icon="📚",
    layout="wide",
)

settings.ensure_directories()


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------


@st.cache_resource
def setup_logging() -> bool:
    """
    Configure PaperLens logging once per Streamlit server process.

    Streamlit re-runs this script on every interaction; caching avoids
    reopening the log file each time.
    """
    configure_logging(
        level=settings.log_level,
        log_file=settings.data_dir / "logs" / "paperlens.jsonl",
    )
    return True


setup_logging()


# ----------------------------------------------------------------------
# Dependency factory
# ----------------------------------------------------------------------


@st.cache_resource
def get_embedder() -> SentenceTransformerEmbedder:
    """Load and cache the embedding model."""
    return SentenceTransformerEmbedder()


@st.cache_resource
def get_llm() -> OllamaLLM:
    """Create and cache the local LLM adapter."""
    return OllamaLLM(settings)


@st.cache_resource
def get_bm25() -> RankBM25Index:
    """Load and cache the persistent BM25 index."""
    index = RankBM25Index(
        persist_path=settings.bm25_store_path,
    )
    index.load()
    return index


@st.cache_resource
def get_image_resolver() -> FilesystemImageResolver:
    """Create and cache the filesystem image resolver."""
    return FilesystemImageResolver(
        allowed_roots=[
            settings.data_dir,
            settings.image_dir,
        ],
    )


@st.cache_resource
def get_vector_lock() -> threading.Lock:
    """
    Local Qdrant allows only one client at a time.

    This lock serializes access when several browser tabs or reruns
    try to open the storage folder concurrently.
    """
    return threading.Lock()


def _build_vector_store() -> QdrantLocalStore:
    """Create a local Qdrant store."""
    return QdrantLocalStore(
        path=settings.vector_store_path,
        dimension=settings.embedding_dimension,
    )


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------


if "history" not in st.session_state:
    st.session_state.history: list[dict[str, Any]] = []


# ----------------------------------------------------------------------
# Application operations
# ----------------------------------------------------------------------


def ask(question: str) -> Answer:
    """Execute the answer-query use case."""

    with get_vector_lock():
        with _build_vector_store() as vector_store:
            use_case = AnswerQueryUseCase(
                settings=settings,
                embedder=get_embedder(),
                vector_store=vector_store,
                bm25_index=get_bm25(),
                image_resolver=get_image_resolver(),
                llm=get_llm(),
            )

            query = Query(
                text=question,
                intent=QueryIntent.TEXT_AND_IMAGE,
            )

            return use_case.run(query)


def get_index_stats() -> tuple[int, int]:
    """Return vector-store and BM25 chunk counts."""

    with get_vector_lock():
        with _build_vector_store() as vector_store:
            vector_count = vector_store.count()

    bm25_count = get_bm25().count()

    return vector_count, bm25_count


# ----------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------


def _render_images(answer: Answer) -> None:
    """Render images attached to the answer."""

    if not answer.used_images:
        return

    columns = st.columns(min(len(answer.used_images), 3))

    for index, image in enumerate(answer.used_images):
        with columns[index % 3]:
            try:
                st.image(
                    image.image_path,
                    caption=f"p.{image.page_number}",
                )
            except Exception as exc:
                logger.warning(
                    "Could not display image %s: %s",
                    image.image_path,
                    exc,
                )
                st.warning(f"Could not load image: {exc}")


def _render_entry(entry: dict[str, Any]) -> None:
    """Render one question/answer exchange as chat bubbles."""

    with st.chat_message("user"):
        st.markdown(entry["question"])

    with st.chat_message("assistant"):
        st.markdown(entry["answer"].text)
        _render_images(entry["answer"])


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------


with st.sidebar:
    st.header("📊 Index")

    try:
        vector_count, bm25_count = get_index_stats()

        st.metric("Vector chunks", vector_count)
        st.metric("BM25 chunks", bm25_count)

    except Exception as exc:
        logger.exception("Could not read index statistics")
        st.error(f"Index unavailable: {exc}")

    st.divider()
    st.header("⚙️ Settings")

    st.write(f"**Model:** `{settings.llm_model}`")
    st.write(f"**Max chunks:** {settings.max_chunks}")

    if st.button("🗑️ Clear conversation"):
        st.session_state.history = []
        st.rerun()


# ----------------------------------------------------------------------
# Main chat
# ----------------------------------------------------------------------


st.title("📚 PaperLens")

if not st.session_state.history:
    st.caption("Ask questions about your research papers.")

# Conversation so far (oldest first, like a normal chat).
for entry in st.session_state.history:
    _render_entry(entry)

# Input pinned at the bottom of the page.
prompt = st.chat_input("Ask a question about your papers...")

if prompt and prompt.strip():
    clean_question = prompt.strip()

    with st.chat_message("user"):
        st.markdown(clean_question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                answer = ask(clean_question)

            st.markdown(answer.text)
            _render_images(answer)

            st.session_state.history.append(
                {
                    "question": clean_question,
                    "answer": answer,
                }
            )

        except Exception as exc:
            logger.exception("Question processing failed")
            st.error(f"Error: {exc}")