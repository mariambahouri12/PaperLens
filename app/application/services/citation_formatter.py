"""Replace [S#] markers by natural, verified source mentions."""

from __future__ import annotations

import re

from app.domain.entities.chunk import Chunk

_MARKER = re.compile(r"\[\s*S(\d+(?:\s*[,;]\s*S?\d+)*)\s*\]")
_NUMBER = re.compile(r"\d+")


def format_source(chunk: Chunk) -> str:
    metadata = chunk.metadata
    parts = [metadata.filename or "document inconnu"]

    pages = list(metadata.page_numbers or ())
    if len(pages) == 1:
        parts.append(f"page {pages[0]}")
    elif pages:
        parts.append("pages " + ", ".join(str(p) for p in pages))

    section = metadata.section_path.as_string()
    if section:
        parts.append(f"section « {section} »")

    return ", ".join(parts)


def apply_citations(
    answer: str,
    chunks: list[Chunk],
) -> tuple[str, list[Chunk]]:
    """
    Return (answer with natural sources, chunks actually cited).
    Markers pointing to unknown chunks are removed.
    """
    cited: dict[int, Chunk] = {}

    def replace(match: re.Match) -> str:
        sources: list[str] = []
        for number in _NUMBER.findall(match.group(1)):
            index = int(number)
            if not 1 <= index <= len(chunks):
                continue
            cited[index] = chunks[index - 1]
            text = format_source(chunks[index - 1])
            if text not in sources:
                sources.append(text)
        if not sources:
            return ""
        return "(Source : " + " ; ".join(sources) + ")"

    text = _MARKER.sub(replace, answer)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)  # espace avant ponctuation
    return text.strip(), [cited[i] for i in sorted(cited)]