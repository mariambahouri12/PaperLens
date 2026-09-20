from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.domain.value_objects.ids import DocumentId, TableId
from app.domain.value_objects.section_path import SectionPath


@dataclass
class ExtractedTable:
    """
    A table extracted as a structured element (not flattened text).
    `rows` is a list of rows; the first row is assumed to hold headers
    when `has_header=True`.
    """

    table_id: TableId
    document_id: DocumentId
    page_number: int
    section_path: SectionPath
    rows: list[list[str]] = field(default_factory=list)
    caption: str = ""
    has_header: bool = True
    bbox: Optional[tuple[float, float, float, float]] = None
    extra: dict = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    def to_markdown(self) -> str:
        if not self.rows:
            return ""
        n_cols = self.n_cols
        padded = [r + [""] * (n_cols - len(r)) for r in self.rows]
        header = "| " + " | ".join(c.replace("|", "\\|") for c in padded[0]) + " |"
        sep = "| " + " | ".join(["---"] * n_cols) + " |"
        body = ["| " + " | ".join(c.replace("|", "\\|") for c in r) + " |" for r in padded[1:]]
        return "\n".join([header, sep, *body])