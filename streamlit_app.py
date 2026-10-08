# streamlit_app.py
"""
PaperLens Streamlit UI — single-file web app.

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from app.application.use_cases.answer_query import AnswerQueryUseCase
from app.config.settings import settings
from app.domain.entities.query import Query, QueryIntent
from app.infrastructure.bm25.rank_bm25_index import RankBM25Index
from app.infrastructure.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from app.infrastructure.image_store.filesystem_image_resolver import (
    FilesystemImageResolver,
)
from app.infrastructure.llm.ollama_llm import OllamaLLM
from app.infrastructure.vector_store.qdrant_local_store import QdrantLocalStore


st.set_page_config(page_title="PaperLens", page_icon="📚", layout="wide")


# ----------------------------------------------------------------------
# Cached resources (loaded once)
# ----------------------------------------------------------------------

@st.cache_resource
def get_embedder():
    return SentenceTransformerEmbedder()


@st.cache_resource
def get_llm():
    return OllamaLLM(settings)


@st.cache_resource
def get_bm25():
    index = RankBM25Index(persist_path=settings.bm25_store_path)
    index.load()
    return index


@st.cache_resource
def get_image_resolver():
    return FilesystemImageResolver(allowed_roots=None)


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# ----------------------------------------------------------------------
# Ask
# ----------------------------------------------------------------------

def ask(question: str):
    with QdrantLocalStore(
        path=settings.vector_store_path,
        dimension=settings.embedding_dimension,
    ) as vector_store:
        use_case = AnswerQueryUseCase(
            settings=settings,
            embedder=get_embedder(),
            vector_store=vector_store,
            bm25_index=get_bm25(),
            image_resolver=get_image_resolver(),
            llm=get_llm(),
        )
        return use_case.run(
            Query(text=question, intent=QueryIntent.TEXT_AND_IMAGE)
        )


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

st.title("📚 PaperLens")
st.caption("Ask questions about your research papers.")

with st.sidebar:
    st.header("📊 Index")
    try:
        with QdrantLocalStore(
            path=settings.vector_store_path,
            dimension=settings.embedding_dimension,
        ) as vs:
            st.metric("Vector chunks", vs.count())
        st.metric("BM25 chunks", get_bm25().count())
    except Exception as exc:
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
    placeholder="e.g. What is FedAvg?",
)

col1, col2 = st.columns([1, 6])
with col1:
    ask_clicked = st.button("🔍 Ask", type="primary", use_container_width=True)

if ask_clicked and question.strip():
    with st.spinner("Thinking..."):
        try:
            answer = ask(question.strip())
            st.session_state.history.insert(0, {
                "question": question.strip(),
                "answer": answer,
            })
        except Exception as exc:
            st.error(f"Error: {exc}")


# ----------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------

for entry in st.session_state.history:
    answer = entry["answer"]

    st.divider()
    st.markdown(f"### ❓ {entry['question']}")
    st.markdown(answer.text)

    if answer.context.chunks:
        with st.expander(f"📖 Sources ({len(answer.context.chunks)})"):
            for chunk in answer.context.chunks:
                meta = chunk.metadata
                section = meta.section_path.as_string() or "root"
                pages = ", ".join(f"p.{p}" for p in meta.page_numbers) or "n/a"
                st.markdown(f"**{meta.filename}** — *{section}* — {pages}")
                st.caption(chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""))
                st.divider()

    if answer.used_images:
        st.markdown(f"#### 🖼️ Figures ({len(answer.used_images)})")
        cols = st.columns(min(len(answer.used_images), 3))
        for i, img in enumerate(answer.used_images):
            with cols[i % 3]:
                try:
                    st.image(img.image_path, caption=f"p.{img.page_number}")
                except Exception as exc:
                    st.warning(f"Could not load image: {exc}")