"""
Sentence splitting that never breaks inside math, LaTeX or abbreviations.
...
"""
from __future__ import annotations

import re

_PROTECTED = re.compile(
    r"""
    \$\$.+?\$\$
  | \$[^$\n]+?\$
  | \\\(.+?\\\)
  | \\\[.+?\\\]
  | \\begin\{(?P<env>[a-zA-Z*]+)\}.+?\\end\{(?P=env)\}
  | \b(?i:et\ al|e\.g|i\.e|etc|vs|cf|figs?|eqs?|tabs?|secs?|refs?|approx|resp)\.
    """,
    re.VERBOSE | re.DOTALL,
)

_PLACEHOLDER = re.compile(r"\x00(\d+)\x00")
_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"“(\[\\\x00])')


def split_sentences(text: str) -> list[str]:
    """Split `text` into sentences without ever cutting inside math."""
    protected: list[str] = []

    def mask(match: re.Match) -> str:
        protected.append(match.group(0))
        return f"\x00{len(protected) - 1}\x00"

    masked = _PROTECTED.sub(mask, text)

    return [
        _PLACEHOLDER.sub(lambda m: protected[int(m.group(1))], part.strip())
        for part in _BOUNDARY.split(masked)
        if part.strip()
    ]