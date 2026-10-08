# image_store/filesystem_image_resolver.py
"""
Filesystem-backed image resolver (read-only).

The extraction pipeline (MinerU + Gemini) already stores images on disk
with content-hash names, and `ChunkMetadata.image_path` holds the real
path. This resolver just validates and reads them — no copy, no
`IMAGE_DIR`, no relative key.

Security: every path is resolved and checked against an optional root
allow-list, so a malformed metadata entry can never read an arbitrary
file from the filesystem.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.domain.exceptions import ImageStoreError
from app.domain.repositories.image_resolver import ImageResolverPort

logger = logging.getLogger("paperlens.infrastructure.image_resolver")


class FilesystemImageResolver(ImageResolverPort):
    """
    Read images directly from the paths stored in chunk metadata.

    `allowed_roots` is an optional list of directories the resolver is
    allowed to read from. When provided, any path outside them is
    rejected. When omitted, no restriction is applied (useful in tests
    and local runs).
    """

    def __init__(self, allowed_roots: list[Path] | None = None) -> None:
        self._roots = tuple(
            root.resolve() for root in (allowed_roots or [])
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def resolve(self, image_path: str) -> Path:
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
        try:
            return self.resolve(image_path).read_bytes()
        except ImageStoreError:
            raise
        except Exception as exc:
            raise ImageStoreError(
                f"Could not load image '{image_path}': {exc}"
            ) from exc

    def exists(self, image_path: str) -> bool:
        try:
            return self.resolve(image_path).is_file()
        except ImageStoreError:
            return False

    # ------------------------------------------------------------------

    def _is_allowed(self, path: Path) -> bool:
        return any(path.is_relative_to(root) for root in self._roots)