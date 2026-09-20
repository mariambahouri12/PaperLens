"""Port: persist and load extracted images."""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.entities.image import ExtractedImage


class ImageStorePort(ABC):
    @abstractmethod
    def save(self, image: ExtractedImage, raw_bytes: bytes) -> None:
        ...

    @abstractmethod
    def load(self, image: ExtractedImage) -> bytes:
        ...

    @abstractmethod
    def exists(self, image_id: str) -> bool:
        ...

    @abstractmethod
    def resolve_path(self, image: ExtractedImage) -> str:
        ...