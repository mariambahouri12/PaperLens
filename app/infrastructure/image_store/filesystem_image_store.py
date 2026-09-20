"""
Filesystem image store.

Writes each image under IMAGE_DIR/<document_id>/<image_id>.png so the
on-disk layout is stable, human-inspectable and easy to back up.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.config.settings import Settings
from app.domain.entities.image import ExtractedImage
from app.domain.exceptions import ImageStoreError
from app.domain.repositories.image_store import ImageStorePort

logger = logging.getLogger("paperlens.infrastructure.image_store")


class FilesystemImageStore(ImageStorePort):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.image_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def save(self, image: ExtractedImage, raw_bytes: bytes) -> None:
        try:
            target = Path(image.image_path) if image.image_path else self._default_path(image)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw_bytes)
            image.image_path = str(target)
            logger.debug("Saved image %s -> %s", image.image_id, target)
        except Exception as exc:
            raise ImageStoreError(f"Could not save image {image.image_id}: {exc}") from exc

    def load(self, image: ExtractedImage) -> bytes:
        try:
            path = Path(image.image_path) if image.image_path else self._default_path(image)
            return path.read_bytes()
        except Exception as exc:
            raise ImageStoreError(f"Could not load image {image.image_id}: {exc}") from exc

    def exists(self, image_id: str) -> bool:
        for doc_dir in self.settings.image_dir.iterdir():
            if doc_dir.is_dir() and (doc_dir / f"{image_id}.png").exists():
                return True
        return False

    def resolve_path(self, image: ExtractedImage) -> str:
        if image.image_path:
            return image.image_path
        return str(self._default_path(image))

    # ------------------------------------------------------------------ #
    def _default_path(self, image: ExtractedImage) -> Path:
        return self.settings.image_dir / image.document_id / f"{image.image_id}.png"