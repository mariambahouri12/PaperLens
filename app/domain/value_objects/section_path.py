"""
A section path is the ordered list of section titles from the root of
the document down to a block, e.g.:

    ["II. BACKGROUND AND RELATED WORK", "A. Federated Learning and FedAvg"]

Kept as a dedicated value object (rather than a bare list) so it can
be serialized consistently, compared, and printed with a stable
"II. BACKGROUND > A. Federated Learning" form.
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

    @property
    def section(self) -> str | None:
        """Top-level section title, if any."""
        return self.parts[0] if self.parts else None

    @property
    def subsection(self) -> str | None:
        """Deepest title below the top-level section, if any."""
        return self.parts[-1] if len(self.parts) > 1 else None

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