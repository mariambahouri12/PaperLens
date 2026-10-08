# domain/entities/chunk.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.domain.value_objects.ids import ChunkId, DocumentId, ImageId, TableId
from app.domain.value_objects.section_path import SectionPath


class ChunkType(str, Enum):
    """Nature of the content of a chunk. Definition order = display order."""

    TEXT = "text"
    EQUATION = "equation"
    LIST = "list"
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"


@dataclass(frozen=True)
class ChunkMetadata:
    """
    Provenance and cross-references of a chunk.

    All chunks of one document share the same `document_id`.
    A text chunk holding equations has types (TEXT, EQUATION).
    Images and tables are referenced by path / id, never embedded.
    """

    document_id: DocumentId
    filename: str
    chunk_index: int
    chunk_types: tuple[ChunkType, ...]
    section_path: SectionPath
    page_numbers: tuple[int, ...] = ()
    image_id: ImageId | None = None
    image_path: str | None = None
    table_id: TableId | None = None

    @property
    def section(self) -> str | None:
        return self.section_path.section

    @property
    def subsection(self) -> str | None:
        return self.section_path.subsection

    def to_flat_dict(self) -> dict:
        """
        Scalar-only metadata for vector stores that reject lists
        (e.g. Chroma). Lists are comma-joined, None values are dropped.
        """
        flat = {
            "document_id": self.document_id,
            "filename": self.filename,
            "chunk_index": self.chunk_index,
            "types": ",".join(t.value for t in self.chunk_types),
            "section": self.section,
            "subsection": self.subsection,
            "section_path": str(self.section_path),
            "pages": ",".join(str(p) for p in self.page_numbers),
            "image_id": self.image_id,
            "image_path": self.image_path,
            "table_id": self.table_id,
        }

        return {key: value for key, value in flat.items() if value is not None}


@dataclass(frozen=True)
class Chunk:
    """
    The retrieval unit of the RAG pipeline: a self-contained,
    section-aware slice of a document, ready to embed and index.
    """

    chunk_id: ChunkId
    document_id: DocumentId
    text: str
    metadata: ChunkMetadata
    token_count: int = 0