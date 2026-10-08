# application/services/context_builder.py
"""
Build the LLM context string from retrieved chunks.

Each chunk is prefixed by a header containing only the fields the LLM
needs to answer source-related questions:

    [document=<filename> | section=<section path> | pages=<page numbers>]

Internal identifiers (`chunk_id`, `document_id`, `chunk_index`, `type`)
are intentionally NOT included: they carry no value for the LLM and
waste context tokens.

Captions (figure / table) are already part of `chunk.text` — the
chunker inlines them — so there is nothing to append.

The caller is responsible for keeping the total under `max_context_tokens`
(`filter_chunks` does this). This function does not truncate.
"""
from __future__ import annotations

from app.domain.entities.chunk import Chunk

_SEPARATOR = "\n\n---\n\n"


def build_context(chunks: list[Chunk]) -> str:
    """Render the chunks as a single string for the LLM prompt."""
    if not chunks:
        return ""

    return _SEPARATOR.join(_render_chunk(chunk) for chunk in chunks)


def _render_chunk(chunk: Chunk) -> str:
    meta = chunk.metadata

    document = meta.filename or "unknown"
    section = meta.section_path.as_string() or "root"
    pages = (
        ", ".join(f"p.{p}" for p in meta.page_numbers)
        if meta.page_numbers
        else "n/a"
    )

    header = f"[document={document} | section={section} | pages={pages}]"

    return f"{header}\n{chunk.text}"