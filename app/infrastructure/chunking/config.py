from __future__ import annotations

from dataclasses import dataclass

from app.config.settings import Settings


@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 512
    overlap_tokens: int = 64

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("overlap_tokens must be in [0, max_tokens)")

    @classmethod
    def from_settings(cls, settings: Settings) -> "ChunkingConfig":
        return cls(
            max_tokens=settings.chunk_size,
            overlap_tokens=settings.chunk_overlap,
        )