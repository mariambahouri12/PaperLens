# domain/repositories/chunker.py
"""Port: split a parsed document into retrieval chunks."""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.entities.chunk import Chunk
from app.domain.entities.document import Document


class ChunkerPort(ABC):
    """
    Split a Document into Chunks.

    Chunk ids must be unique within a document and stable across runs
    (same document + same chunker config => same chunk ids), so the
    checkpoint can safely skip already-indexed chunks.
    """

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Return the chunks in reading order. Chunks never cross a
        section boundary, so their section metadata is always exact."""