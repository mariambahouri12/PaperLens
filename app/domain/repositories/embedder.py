"""Port: produce dense vector embeddings from text."""
from __future__ import annotations

from abc import ABC, abstractmethod


class EmbedderPort(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        ...