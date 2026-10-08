# domain/repositories/checkpoint.py
"""Port: resumable ingestion checkpoint."""
from __future__ import annotations

from abc import ABC, abstractmethod


class CheckpointRepository(ABC):
    """
    Tracks which chunks have been fully indexed (vector store AND BM25,
    both persisted).

    A chunk must be marked completed ONLY after every index that needs
    it has accepted it; otherwise a crash leaves the two indexes out of
    sync and the missing chunks are never retried.
    """

    @abstractmethod
    def is_completed(self, chunk_id: str) -> bool:
        """True if the chunk was successfully indexed in a previous run."""

    @abstractmethod
    def mark_completed(self, chunk_ids: list[str]) -> None:
        """Persist the given ids as completed. Must be atomic (a crash
        during the write must not corrupt the file)."""