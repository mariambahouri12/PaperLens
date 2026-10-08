# domain/repositories/image_resolver.py
"""
Read-only access to image files referenced by chunks.

Images are NEVER copied: the extraction pipeline already stores them on
disk (MinerU writes content-hash file names) and the chunk metadata
carries the real path. This port only exposes a safe way to read them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ImageResolverPort(ABC):
    """Resolve and load images referenced by chunks."""

    @abstractmethod
    def resolve(self, image_path: str) -> Path:
        """Absolute path of the image.

        Raises ImageResolverError if the path is empty, outside the
        allowed roots, or does not point to a readable file.
        """

    @abstractmethod
    def load(self, image_path: str) -> bytes:
        """Raw bytes of the image.

        Raises ImageResolverError on any failure.
        """

    @abstractmethod
    def exists(self, image_path: str) -> bool:
        """True if the image file exists and is readable."""