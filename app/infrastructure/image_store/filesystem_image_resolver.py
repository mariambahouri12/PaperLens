"""
Filesystem-backed image resolver (read-only).

The extraction pipeline (MinerU + Gemini) already stores images on disk
with content-hash names, and `ChunkMetadata.image_path` holds the real
path. This resolver validates and reads those paths — no copy and no
hidden image directory.

Security: every path is resolved and checked against an optional root
allow-list, so a malformed metadata entry cannot read a file outside
the configured roots.
"""
from __future__ import annotations

from pathlib import Path

from app.domain.exceptions import ImageStoreError
from app.domain.repositories.image_resolver import ImageResolverPort


class FilesystemImageResolver(ImageResolverPort):
    """
    Read images directly from paths stored in chunk metadata.

    `allowed_roots` is an optional list of directories the resolver is
    allowed to read from. When provided, paths outside those roots are
    rejected.
    """

    def __init__(self, allowed_roots: list[Path] | None = None) -> None:
        self._roots = tuple(
            root.resolve() for root in (allowed_roots or [])
        )

    def resolve(self, image_path: str) -> Path:
        """
        Validate an image path and return its resolved absolute path.

        Raises:
            ImageStoreError: If the path is empty, invalid, outside the
                allowed roots, or does not point to a file.
        """
        if not image_path:
            raise ImageStoreError("Empty image path")

        try:
            path = Path(image_path).resolve()
        except (OSError, ValueError) as exc:
            raise ImageStoreError(
                f"Invalid image path {image_path!r}: {exc}"
            ) from exc

        if self._roots and not self._is_allowed(path):
            raise ImageStoreError(
                f"Image path {image_path!r} is outside the allowed roots"
            )

        if not path.is_file():
            raise ImageStoreError(f"Image not found: {image_path}")

        return path

    def load(self, image_path: str) -> bytes:
        """
        Read and return the raw bytes of an image.

        Raises:
            ImageStoreError: If the path is invalid or the file cannot
                be read.
        """
        path = self.resolve(image_path)

        try:
            return path.read_bytes()
        except OSError as exc:
            raise ImageStoreError(
                f"Could not load image '{image_path}': {exc}"
            ) from exc

    def exists(self, image_path: str) -> bool:
        """
        Return whether the image path is valid, allowed, and points
        to an existing file.
        """
        try:
            self.resolve(image_path)
        except ImageStoreError:
            return False

        return True

    def _is_allowed(self, path: Path) -> bool:
        """Return whether the resolved path belongs to an allowed root."""
        return any(
            path.is_relative_to(root)
            for root in self._roots
        )
