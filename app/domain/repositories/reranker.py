"""Port: rerank retrieved chunks for a query."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

SearchResult = tuple[str, float, dict[str, Any]]


class RerankerPort(ABC):
    """Rerank candidates using query–document relevance."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Return the top candidates in reranked order."""