"""
Content blocks of a parsed document.

A block is the smallest unit of content, as read from the paper:
a paragraph, an equation, a list, a table or a visual (image / chart).
Blocks are immutable value-like entities and know nothing about
chunking or about the tool that extracted them.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.domain.value_objects.ids import ImageId, TableId


class VisualKind(str, Enum):
    IMAGE = "image"
    CHART = "chart"


@dataclass(frozen=True)
class Block:
    page: int | None


@dataclass(frozen=True)
class TextBlock(Block):
    text: str


@dataclass(frozen=True)
class EquationBlock(Block):
    latex: str


@dataclass(frozen=True)
class ListBlock(Block):
    items: tuple[str, ...]


@dataclass(frozen=True)
class VisualBlock(Block):
    """An image or a chart. The binary stays on disk, referenced by path."""

    kind: VisualKind
    image_id: ImageId
    image_path: str | None
    caption: str = ""
    footnote: str = ""


@dataclass(frozen=True)
class TableBlock(Block):
    """A table kept as a structured grid (not flattened text)."""

    table_id: TableId
    columns: tuple[str, ...]
    rows: tuple[tuple[str | None, ...], ...]
    caption: str = ""
    footnote: str = ""
    image_path: str | None = None

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return len(self.columns)

    def to_markdown(self) -> str:
        """Markdown grid of the table (without caption / footnote)."""
        if not self.columns:
            return ""

        def line(cells) -> str:
            escaped = ((c or "").replace("|", "\\|").replace("\n", " ") for c in cells)
            return "| " + " | ".join(escaped) + " |"

        padded = [
            (list(row) + [None] * self.n_cols)[: self.n_cols]
            for row in self.rows
        ]

        return "\n".join(
            [line(self.columns), line(["---"] * self.n_cols), *map(line, padded)]
        )