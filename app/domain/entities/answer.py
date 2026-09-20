from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.entities.chunk import Chunk
from app.domain.entities.image import ExtractedImage


@dataclass
class RetrievedContext:
    chunks: list[Chunk] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)
    total_tokens: int = 0


@dataclass
class Answer:
    text: str
    context: RetrievedContext
    used_images: list[ExtractedImage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)