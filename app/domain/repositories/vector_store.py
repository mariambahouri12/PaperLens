# domain/repositories/vector_store.py
"""Port: store and query dense vectors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any

SearchResult = tuple[str, float, dict[str, Any]]


class VectorStorePort(ABC):
    """
    Dense vector index over chunk embeddings.

    Implementations may hold OS-level resources (file locks, sockets).
    They MUST release them on `close()`, and SHOULD support being used
    as a context manager so the composition root never leaks them.
    """

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """Insert or replace points. All three lists must have the same
        length; a mismatch must raise IndexingError."""

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int,
    ) -> list[SearchResult]:
        """Return (chunk_id, similarity_score, payload) sorted by score
        descending. Must raise IndexingError on storage failure, never
        silently return []."""

    @abstractmethod
    def exists(self, chunk_id: str) -> bool:
        """True if a point with this chunk_id is stored."""

    @abstractmethod
    def count(self) -> int:
        """Number of points currently stored."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def close(self) -> None:
        """Release the storage lock and flush to disk. Idempotent."""

    def __enter__(self) -> "VectorStorePort":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()