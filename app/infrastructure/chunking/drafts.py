from __future__ import annotations

from dataclasses import dataclass

from app.domain.entities.chunk import ChunkType
from app.domain.value_objects.ids import ImageId, TableId


def order_types(types) -> tuple[ChunkType, ...]:
    """Unique chunk types, in the enum's definition order."""
    present = set(types)
    return tuple(t for t in ChunkType if t in present)


@dataclass(frozen=True)
class ChunkDraft:
    """A chunk before it receives its id and section metadata."""

    text: str
    types: tuple[ChunkType, ...]
    pages: tuple[int, ...]
    image_id: ImageId | None = None
    image_path: str | None = None
    table_id: TableId | None = None