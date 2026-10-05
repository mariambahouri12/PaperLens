#extraction/document_mapper.py
"""
Maps the JSON produced by the extraction pipeline to a domain Document.

This is the only place that knows the JSON layout. Layout noise emitted
by MinerU is dropped here, so the domain and the chunker never see it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Iterator

from app.domain.entities.block import (
    Block,
    EquationBlock,
    ListBlock,
    TableBlock,
    TextBlock,
    VisualBlock,
    VisualKind,
)
from app.domain.entities.document import Document, DocumentMetadata
from app.domain.entities.section import Section
from app.domain.value_objects.ids import DocumentId, ImageId, TableId
from app.domain.value_objects.section_path import SectionPath

# Adapt this import to where front_matter.py lives in your project.
from app.infrastructure.extraction.front_matter import (
    ABSTRACT_TITLE,
    AUTHORS_TITLE,
)

IGNORED_TYPES = frozenset(
    {"aside_text", "header", "footer", "page_number", "page_footnote"}
)


def _as_text(value) -> str:
    """Normalize a str, a list of str, or None into a single string."""
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item).strip()

    return str(value or "").strip()


@dataclass
class _Counters:
    """Fallback id generators for assets that have no id in the JSON."""

    images: Iterator[int] = field(default_factory=lambda: count(1))
    tables: Iterator[int] = field(default_factory=lambda: count(1))


class DocumentMapper:
    def to_document(
        self,
        data: dict,
        document_id: DocumentId,
        file_path: Path | None = None,
    ) -> Document:
        meta = data.get("metadata") or {}
        root = self._map_section(
            data.get("content") or {},
            SectionPath.empty(),
            _Counters(),
        )

        document = Document(
            metadata=DocumentMetadata(
                document_id=document_id,
                filename=meta.get("source_file") or "",
                file_path=str(file_path) if file_path else "",
                file_size_bytes=(
                    file_path.stat().st_size
                    if file_path and file_path.is_file()
                    else 0
                ),
                page_count=int(meta.get("pages") or 0),
                title=meta.get("title_guess"),
            ),
            root=root,
        )

        self._fill_front_matter(document)

        return document

    # ------------------------------------------------------------------

    def _map_section(
        self,
        raw: dict,
        parent_path: SectionPath,
        counters: _Counters,
    ) -> Section:
        title = raw.get("title")
        path = parent_path.child(title) if title else parent_path

        blocks = [
            block
            for block in (
                self._map_block(item, counters)
                for item in raw.get("blocks") or []
            )
            if block is not None
        ]

        return Section(
            title=title,
            level=int(raw.get("level") or 0),
            page=raw.get("page"),
            path=path,
            blocks=blocks,
            subsections=[
                self._map_section(sub, path, counters)
                for sub in raw.get("subsections") or []
            ],
        )

    def _map_block(self, raw: dict, counters: _Counters) -> Block | None:
        kind = raw.get("type")
        page = raw.get("page")

        if kind in IGNORED_TYPES:
            return None

        if kind == "equation":
            latex = _as_text(raw.get("latex") or raw.get("text"))
            return EquationBlock(page=page, latex=latex) if latex else None

        if kind == "list":
            items = tuple(
                text
                for text in map(_as_text, raw.get("list_items") or [])
                if text
            )
            return ListBlock(page=page, items=items) if items else None

        if kind == "table":
            return self._map_table(raw, page, counters)

        if kind in ("image", "chart"):
            return self._map_visual(raw, kind, page, counters)

        # "text" and any unknown type: keep it only if it carries text.
        text = _as_text(raw.get("text"))

        return TextBlock(page=page, text=text) if text else None

    @staticmethod
    def _map_table(raw: dict, page, counters: _Counters) -> TableBlock:
        return TableBlock(
            page=page,
            table_id=TableId(raw.get("id") or f"table_{next(counters.tables)}"),
            columns=tuple(str(c) for c in raw.get("columns") or []),
            rows=tuple(
                tuple(None if cell is None else str(cell) for cell in row)
                for row in raw.get("rows") or []
            ),
            caption=_as_text(raw.get("caption")),
            footnote=_as_text(raw.get("footnote")),
            image_path=raw.get("image_path") or None,
        )

    @staticmethod
    def _map_visual(raw: dict, kind: str, page, counters: _Counters) -> VisualBlock:
        return VisualBlock(
            page=page,
            kind=VisualKind(kind),
            image_id=ImageId(f"image_{next(counters.images)}"),
            image_path=raw.get("image_path") or raw.get("img_path") or None,
            caption=_as_text(raw.get("caption") or raw.get("chart_caption")),
            footnote=_as_text(raw.get("footnote") or raw.get("chart_footnote")),
        )

    @staticmethod
    def _fill_front_matter(document: Document) -> None:
        """Expose authors / abstract in the metadata when they were found."""
        authors = document.find_section(AUTHORS_TITLE)
        abstract = document.find_section(ABSTRACT_TITLE)

        if authors:
            document.metadata.author_lines = [
                b.text for b in authors.blocks if isinstance(b, TextBlock)
            ]

        if abstract:
            first = next(
                (b for b in abstract.blocks if isinstance(b, TextBlock)),
                None,
            )
            document.metadata.abstract = first.text if first else None