"""
Build the LLM context from retrieved domain chunks.

Only information useful to the LLM is exposed in the context header.
Internal identifiers are intentionally excluded.
"""

from __future__ import annotations

from app.domain.entities.chunk import Chunk

_SEPARATOR = "\n\n---\n\n"


def build_context(
    chunks: list[Chunk],
) -> str:
    """Render chunks into the context sent to the LLM."""

    if not chunks:
        return ""

    return _SEPARATOR.join(
        _render_chunk(chunk)
        for chunk in chunks
    )


def _render_chunk(
    chunk: Chunk,
) -> str:
    metadata = chunk.metadata

    document = metadata.filename or "unknown"

    section = (
        metadata.section_path.as_string()
        or "root"
    )

    pages = (
        ", ".join(
            f"p.{page}"
            for page in metadata.page_numbers
        )
        if metadata.page_numbers
        else "n/a"
    )

    header = (
        f"[document={document} | "
        f"section={section} | "
        f"pages={pages}]"
    )

    return f"{header}\n{chunk.text}"