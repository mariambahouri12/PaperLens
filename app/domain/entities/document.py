from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from app.domain.value_objects.ids import DocumentId


@dataclass
class DocumentMetadata:
    document_id: DocumentId
    filename: str
    file_path: str
    file_size_bytes: int
    page_count: int
    title: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    abstract: Optional[str] = None
    ingested_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Document:
    """
    A parsed research paper.

    Holds the metadata and the flat, reading-ordered list of elements
    (paragraphs, headings, figures, tables, captions, ...) produced by
    the extraction layer. The hierarchical section tree is derived from
    the elements by the chunker, not stored here.
    """

    metadata: DocumentMetadata
    elements: list["DocumentElement"] = field(default_factory=list)
    images: list["ExtractedImage"] = field(default_factory=list)
    tables: list["ExtractedTable"] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Late imports to avoid cycles at module load
from app.domain.entities.element import DocumentElement  # noqa: E402
from app.domain.entities.image import ExtractedImage  # noqa: E402
from app.domain.entities.table import ExtractedTable  # noqa: E402