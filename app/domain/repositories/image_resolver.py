
"""
Read-only access to image files referenced by chunks.

Images are never copied: the extraction pipeline already stores them on
disk, and chunk metadata carries the real path. This port exposes only
the operations required to safely resolve and read those images.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ImageResolverPort(ABC):
    """Port for resolving and loading images referenced by chunks."""

    @abstractmethod
    def resolve(self, image_path: str) -> Path:
        """
        Return the resolved absolute path of an image.

        Raises:
            ImageStoreError: If the path is empty, invalid, outside the
                allowed roots, or does not point to a readable file.
        """

    @abstractmethod
    def load(self, image_path: str) -> bytes:
        """
        Return the raw bytes of an image.

        Raises:
            ImageStoreError: If the image cannot be loaded.
        """

    @abstractmethod
    def exists(self, image_path: str) -> bool:
        """
        Return whether the image exists and is accessible through
        the resolver.
        """
