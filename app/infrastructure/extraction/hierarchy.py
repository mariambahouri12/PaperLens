from __future__ import annotations

import re

_LETTER = re.compile(r"^[A-HJ-UWYZ]\.\s+")
_ROMAN = re.compile(r"^[IVXLCDM]+\.\s+")
_DECIMAL = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+")


def normalize_title(title: str) -> str:
    title = str(title or "")

    for space in ("\u00a0", "\u2007", "\u202f"):
        title = title.replace(space, " ")

    return re.sub(r"\s+", " ", title).strip()


def has_section_number(title: str) -> bool:
    title = normalize_title(title)

    return bool(
        _LETTER.match(title)
        or _ROMAN.match(title)
        or _DECIMAL.match(title)
    )


def detect_logical_level(title: str) -> int:
    """
    Detect the logical hierarchy level of a heading.

        A. / B. / C.        -> level 2 (I, V, X exclus : chiffres romains)
        I. / II. / V. / X.  -> level 1
        1. / 2.1 / 2.1.1    -> level 1 / 2 / 3
        Unnumbered heading  -> level 1
    """
    title = normalize_title(title)

    if _LETTER.match(title):
        return 2

    if _ROMAN.match(title):
        return 1

    decimal_match = _DECIMAL.match(title)

    if decimal_match:
        return len(decimal_match.group(1).split("."))

    return 1