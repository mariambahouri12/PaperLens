from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.value_objects.ids import ChunkId, DocumentId, ImageId, TableId
from app.domain.value_objects.section_path import SectionPath


@dataclass
class ChunkMetadata:
    """
    Metadata attached to every chunk. Preserves provenance (document,
    page, section), and cross-references to first-class assets
    (images, tables) so retrieval can resolve them without duplicating
    binary content inside the chunk text.
    """

    document_id: DocumentId
    filename: str
    section_path: SectionPath
    page_numbers: list[int] = field(default_factory=list)
    image_ids: list[ImageId] = field(default_factory=list)
    image_captions: list[str] = field(default_factory=list)
    table_ids: list[TableId] = field(default_factory=list)
    table_captions: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """
    The retrieval unit of the RAG pipeline.

    A chunk is a self-contained, section-aware slice of the document,
    ready to be embedded, indexed, retrieved and passed to the LLM.
    """

    chunk_id: ChunkId
    document_id: DocumentId
    text: str
    metadata: ChunkMetadata
    token_count: int = 0