
"""
Port: lexical BM25 index over chunk texts.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


SearchResult = tuple[str, float, dict[str, Any]]


class BM25IndexPort(ABC):
    """
    Lexical index over chunk texts.

    The index is in-memory; `persist()` flushes it to disk and `load()`
    restores it. Unlike the vector store, BM25 has no OS-level lock, so
    `close()` is a no-op — kept for uniformity with VectorStorePort.
    """

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @abstractmethod
    def add(
        self,
        chunk_ids: list[str],
        texts: list[str],
        payloads: list[dict[str, Any]],
    ) -> None:
        """Add chunks. Existing IDs are ignored (idempotent)."""

    @abstractmethod
    def persist(self) -> None:
        """Write the current state to disk atomically."""

    @abstractmethod
    def load(self) -> None:
        """
        Restore the state from disk.

        Must raise IndexingError if the file exists but is unreadable.
        """

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        """
        Return (chunk_id, BM25 score, payload) sorted by score descending.

        Must raise IndexingError on failure, never silently return [].
        """

    @abstractmethod
    def count(self) -> int:
        """Return the number of documents currently indexed."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        No-op for BM25; kept for interface uniformity.

        The method is idempotent.
        """
