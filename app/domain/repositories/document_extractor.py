# domain/repositories/document_extractor.py
"""Port: extract structure and assets from a raw document file."""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.entities.document import Document


class DocumentExtractorPort(ABC):
    """
    Parse a raw file (PDF, DOCX, ...) into a domain Document.

    Implementations are format-specific (one per supported extension);
    the ingestion use case picks the first extractor whose `supports`
    returns True.
    """

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """True if this extractor can handle the file (usually by
        extension)."""

    @abstractmethod
    def extract(self, file_path: str) -> Document:
        """Return a fully parsed Document (section tree with blocks:
        text, equations, lists, tables, images, charts).

        Images are NOT loaded in memory: their `image_path` is recorded
        in the corresponding VisualBlock. Raises ExtractionError on
        unrecoverable failures.
        """