#domain/sentities/section.py
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from app.domain.entities.block import Block
from app.domain.value_objects.section_path import SectionPath


@dataclass
class Section:
    """
    A node of the document's section hierarchy.

    A section owns its direct blocks and its subsections. `path` is the
    full path of titles from the root, so every block can be located
    without storing the path on the block itself.
    The root section has no title and an empty path.
    """

    title: str | None
    level: int
    page: int | None
    path: SectionPath
    blocks: list[Block] = field(default_factory=list)
    subsections: list["Section"] = field(default_factory=list)

    def walk(self) -> Iterator["Section"]:
        """Yield this section then all descendants, in reading order."""
        yield self

        for subsection in self.subsections:
            yield from subsection.walk()