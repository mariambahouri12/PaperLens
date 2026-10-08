"""
Port for storing and querying dense vectors.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any

SearchResult = tuple[str, float, dict[str, Any]]


class VectorStorePort(ABC):
    """
    Dense vector index over chunk embeddings.

    Implementations may hold OS-level resources such as file locks or
    sockets. They must release those resources through `close()` and
    should support context-manager usage.
    """

    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """
        Insert or replace points.

        All three lists must have the same length. A mismatch must raise
        IndexingError.
        """

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Return (chunk_id, similarity_score, payload) sorted by decreasing
        score.

        Storage failures must raise IndexingError rather than silently
        returning an empty result.
        """

    @abstractmethod
    def exists(self, chunk_id: str) -> bool:
        """
        Return whether a point for the given chunk ID is stored.

        Storage failures must raise IndexingError rather than being
        reported as a missing point.
        """

    @abstractmethod
    def count(self) -> int:
        """
        Return the number of points currently stored.

        Storage failures must raise IndexingError rather than returning
        a potentially misleading zero.
        """

    @abstractmethod
    def close(self) -> None:
        """
        Release storage resources.

        The operation must be idempotent.
        """

    def __enter__(self) -> "VectorStorePort":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
