"""Port: store and query dense vectors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VectorStorePort(ABC):
    @abstractmethod
    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        ...

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float, dict[str, Any]]]:
        """Return (chunk_id, similarity_score, payload) sorted by score
        descending."""

    @abstractmethod
    def exists(self, chunk_id: str) -> bool:
        ...

    @abstractmethod
    def count(self) -> int:
        ...