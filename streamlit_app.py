"""
PaperLens Streamlit UI.

Run:
    streamlit run streamlit_app.py

The Streamlit layer is responsible only for:
    - UI rendering
    - session state
    - composition of application dependencies

Business and retrieval logic remain in the application/domain layers.
"""

from __future__ import annotations

import logging
import re
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
from app.infrastructure.vector_store.qdrant_local_store import (
    QdrantLocalStore,
)

logger = logging.getLogger("paperlens.interfaces.streamlit")

st.set_page_config(
    page_title="PaperLens",
    page_icon="📚",
    layout="wide",
)


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

    with _build_vector_store() as vector_store:
        vector_count = vector_store.count()

    bm25_count = get_bm25().count()

    return vector_count, bm25_count


# ----------------------------------------------------------------------
# Source relevance
# ----------------------------------------------------------------------


def _normalize_terms(text: str) -> set[str]:
    """Normalize text into lowercase alphanumeric terms."""

    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


def _select_relevant_sources(
    answer: Answer,
    question: str,
    max_sources: int = 2,
) -> list[Any]:
    """
    Select a small number of source chunks using a simple lexical score.

    Section-name matches receive more weight than body-text matches.
    This is a lightweight heuristic, not a semantic reranker.
    """

    if not answer.context.chunks:
        return []

    stop_words = {
        "the", "of", "is", "are", "was", "were",
        "what", "who", "which", "where", "when", "how",
        "a", "an", "and", "or", "to", "in", "on", "for",
        "about", "from", "with", "this", "that", "these",
        "those", "paper", "please", "explain", "tell",
    }

    question_terms = _normalize_terms(question) - stop_words

    # Avoid matching generic words such as "authors" only against
    # the document section title.
    if not question_terms:
        question_terms = _normalize_terms(question)

    ranked: list[tuple[int, int, Any]] = []

    for index, chunk in enumerate(answer.context.chunks):
        metadata = chunk.metadata
        section = metadata.section_path.as_string() or ""

        section_terms = _normalize_terms(section)
        filename_terms = _normalize_terms(metadata.filename or "")
        body_terms = _normalize_terms(chunk.text)

        section_matches = len(question_terms & section_terms)
        filename_matches = len(question_terms & filename_terms)
        body_matches = len(question_terms & body_terms)

        score = (
            4 * section_matches
            + 2 * filename_matches
            + body_matches
        )

        ranked.append((score, index, chunk))

    # Keep original retrieval order as a tie-breaker.
    ranked.sort(key=lambda item: (-item[0], item[1]))

    selected = []
    seen_sources = set()

    for score, _, chunk in ranked:
        if score <= 0:
            continue

        metadata = chunk.metadata

        source_key = (
            metadata.filename or "unknown",
            tuple(metadata.page_numbers),
            metadata.section_path.as_string() or "root",
        )

        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)
        selected.append(chunk)

        if len(selected) >= max_sources:
            break

    # Fallback: if there are no lexical matches, use the top-ranked
    # retrieved chunk rather than displaying every retrieved chunk.
    if not selected:
        selected.append(answer.context.chunks[0])

    return selected


# ----------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------


def _render_sources(
    answer: Answer,
    question: str,
) -> None:
    """Render concise source metadata without chunk previews."""

    sources = _select_relevant_sources(
        answer=answer,
        question=question,
        max_sources=2,
    )

    if not sources:
        return

    st.markdown(
        "**Source:**" if len(sources) == 1 else "**Sources:**"
    )

    for chunk in sources:
        metadata = chunk.metadata

        document = metadata.filename or "unknown"
        section = metadata.section_path.as_string() or "root"

        pages = (
            ", ".join(f"p.{page}" for page in metadata.page_numbers)
            if metadata.page_numbers
            else "p.n/a"
        )

        st.caption(
            f"📄 {document} — {pages} — section {section}"
        )


def _render_images(answer: Answer) -> None:
    """Render images attached to the answer."""

    if not answer.used_images:
        return

    st.markdown(f"#### 🖼️ Figures ({len(answer.used_images)})")

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


def _render_answer(entry: dict[str, Any]) -> None:
    """Render one question/answer entry."""

    answer: Answer = entry["answer"]
    question = entry["question"]

    st.divider()
    st.markdown(f"### ❓ {question}")
    st.markdown(answer.text)

    _render_sources(
        answer=answer,
        question=question,
    )

    _render_images(answer)


# ----------------------------------------------------------------------
# Main UI
# ----------------------------------------------------------------------


st.title("📚 PaperLens")
st.caption("Ask questions about your research papers.")

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

    if st.button("🗑️ Clear history"):
        st.session_state.history = []
        st.rerun()


question = st.text_input(
    "Your question",
    placeholder="e.g. Who are the authors of the paper?",
)

col1, col2 = st.columns([1, 6])

with col1:
    ask_clicked = st.button(
        "🔍 Ask",
        type="primary",
        use_container_width=True,
    )


if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        clean_question = question.strip()

        with st.spinner("Thinking..."):
            try:
                answer = ask(clean_question)

                st.session_state.history.insert(
                    0,
                    {
                        "question": clean_question,
                        "answer": answer,
                    },
                )

            except Exception as exc:
                logger.exception("Question processing failed")
                st.error(f"Error: {exc}")


# ----------------------------------------------------------------------
# History
# ----------------------------------------------------------------------


for entry in st.session_state.history:
    _render_answer(entry)
