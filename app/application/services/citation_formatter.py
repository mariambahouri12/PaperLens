"""Replace [S#] markers by natural, verified source mentions."""

from __future__ import annotations

import re

from app.domain.entities.chunk import Chunk

# One marker group: [S1], [S1, S2], [S3 | S5], [S1; 2], [S1 and S2], [S1 S2].
_ITEM = r"S?\d+"
_SEP = r"(?:\s*[,;|/&]\s*|\s+(?:and|et)\s+|\s+)"
_GROUP = rf"\[\s*S\d+(?:{_SEP}{_ITEM})*\s*\]"

# A run of adjacent groups, e.g. "[S1][S3]" or "[S1] [S2]": they are
# merged into a single "(Source : ...)" mention.
_RUN = re.compile(rf"{_GROUP}(?:\s*{_GROUP})*", re.IGNORECASE)

# The LLM sometimes copies a whole context header such as
# "[S2 | document=a.pdf | section=... | pages=1]" instead of "[S2]".
_HEADER_ECHO = re.compile(
    r"\[\s*S(\d+)\s*\|\s*document\s*=[^\[\]]*\]",
    re.IGNORECASE,
)

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

    Handled marker shapes: [S1], [S1, S2], [S3 | S5], [S1][S2] and a
    copied context header. Markers pointing to unknown chunks are removed.
    Bibliographic references such as "[10]" are left untouched because
    markers must start with "S".
    """
    cited: dict[int, Chunk] = {}

    def replace(match: re.Match) -> str:
        sources: list[str] = []

        for number in _NUMBER.findall(match.group(0)):
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

    text = _HEADER_ECHO.sub(lambda m: f"[S{m.group(1)}]", answer)
    text = _RUN.sub(replace, text)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)  # espace avant ponctuation
    text = re.sub(r"(?<=\S)[ \t]{2,}", " ", text)  # espaces doubles laissés

    return text.strip(), [cited[i] for i in sorted(cited)]