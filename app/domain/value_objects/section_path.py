"""
A section path is the ordered list of section titles from the root of
the document down to the element, e.g.:

    ["Bayesian Methods", "Results", "Experimental Results"]

Kept as a dedicated value object (rather than a bare list) so it can
be serialized consistently, compared, and printed with a stable
"Bayesian Methods > Results > Experimental Results" form.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SectionPath:
    parts: tuple[str, ...]

    @classmethod
    def empty(cls) -> "SectionPath":
        return cls(())

    @classmethod
    def of(cls, parts: Iterable[str]) -> "SectionPath":
        return cls(tuple(p for p in parts if p))

    def child(self, title: str) -> "SectionPath":
        return SectionPath(self.parts + (title,))

    def parent(self) -> "SectionPath":
        return SectionPath(self.parts[:-1])

    def as_list(self) -> list[str]:
        return list(self.parts)

    def as_string(self, sep: str = " > ") -> str:
        return sep.join(self.parts)

    def __bool__(self) -> bool:
        return bool(self.parts)

    def __str__(self) -> str:
        return self.as_string()