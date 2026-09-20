"""Port: lexical BM25 index over chunk texts."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BM25IndexPort(ABC):
    @abstractmethod
    def add(self, chunk_ids: list[str], texts: list[str], payloads: list[dict[str, Any]]) -> None:
        ...

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[tuple[str, float, dict[str, Any]]]:
        """Return (chunk_id, bm25_score, payload) sorted by score
        descending."""

    @abstractmethod
    def persist(self) -> None:
        ...

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def count(self) -> int:
        ...