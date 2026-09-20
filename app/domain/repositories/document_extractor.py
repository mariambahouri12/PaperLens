"""Port: extract structure and assets from a raw document file."""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.entities.document import Document


class DocumentExtractorPort(ABC):
    @abstractmethod
    def supports(self, file_path: str) -> bool:
        ...

    @abstractmethod
    def extract(self, file_path: str) -> Document:
        """Return a fully parsed Document (elements, images, tables,
        warnings). Raises ExtractionError on unrecoverable failures."""