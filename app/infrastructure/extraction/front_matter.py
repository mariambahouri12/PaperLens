#extraction/front_matter.py
from __future__ import annotations

import re

ABSTRACT_TITLE = "ABSTRACT"
AUTHORS_TITLE = "Authors"

# Matches "Abstract—...", "Abstract: ...", "ABSTRACT. ...", "Abstract ...".
_ABSTRACT_PREFIX = re.compile(
    r"^\s*abstract\b\s*[:.\-–—]*\s*",
    re.IGNORECASE,
)


def split_abstract(text: str) -> str | None:
    """
    If the text starts with "Abstract", return the body without the prefix
    (an empty string if the word stands alone). Otherwise return None.
    """
    match = _ABSTRACT_PREFIX.match(text or "")

    if match is None:
        return None

    return text[match.end():].strip()