"""Port: turn a parsed Document into retrieval Chunks."""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.entities.chunk import Chunk
from app.domain.entities.document import Document


class ChunkerPort(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        ...