"""
Build the LLM context from retrieved domain chunks.

Each chunk is prefixed with a marker [S1], [S2], ... that the LLM uses
to indicate which chunk supports a claim. The application then replaces
the markers with verified, natural source mentions.

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
        _render_chunk(chunk, index)
        for index, chunk in enumerate(chunks, start=1)
    )


def _render_chunk(
    chunk: Chunk,
    index: int,
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
        f"[S{index} | document={document} | "
        f"section={section} | "
        f"pages={pages}]"
    )

    return f"{header}\n{chunk.text}"