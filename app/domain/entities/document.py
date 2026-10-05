from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from app.domain.entities.block import TableBlock
from app.domain.entities.section import Section
from app.domain.value_objects.ids import DocumentId


@dataclass
class DocumentMetadata:
    document_id: DocumentId
    filename: str
    file_path: str
    file_size_bytes: int
    page_count: int
    title: Optional[str] = None
    # Raw front-matter lines (names, affiliations, emails), as printed.
    author_lines: list[str] = field(default_factory=list)
    abstract: Optional[str] = None
    ingested_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class Document:
    """
    A parsed research paper: metadata plus its section tree.

    The tree (root -> sections -> subsections -> blocks) preserves the
    reading order. Chunkers walk it; they never see extraction details.
    """

    metadata: DocumentMetadata
    root: Section
    warnings: list[str] = field(default_factory=list)

    def iter_sections(self) -> Iterator[Section]:
        """All sections in reading order (the untitled root comes first)."""
        yield from self.root.walk()

    def iter_tables(self) -> Iterator[TableBlock]:
        for section in self.iter_sections():
            for block in section.blocks:
                if isinstance(block, TableBlock):
                    yield block

    def find_section(self, title: str) -> Section | None:
        return next(
            (s for s in self.iter_sections() if s.title == title),
            None,
        )

    def table_of_contents(self) -> list[tuple[int, str, int | None]]:
        return [
            (s.level, s.title, s.page)
            for s in self.iter_sections()
            if s.title and s.level > 0
        ]