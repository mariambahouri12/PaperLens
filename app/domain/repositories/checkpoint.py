
"""
Port for resumable ingestion checkpoints.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class CheckpointRepository(ABC):
    """
    Tracks chunks that have been fully indexed.

    A chunk must only be marked as completed after every required index
    has successfully accepted and persisted it.
    """

    @abstractmethod
    def is_completed(self, chunk_id: str) -> bool:
        """
        Return whether the chunk was successfully indexed previously.
        """

    @abstractmethod
    def mark_completed(self, chunk_ids: list[str]) -> None:
        """
        Persist the given chunk IDs as completed.

        Implementations must ensure that a checkpoint write cannot leave
        a partially written checkpoint file after a crash.
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Clear all completed chunk IDs and persist the empty checkpoint.
        """

    @abstractmethod
    def count(self) -> int:
        """
        Return the number of completed chunk IDs.
        """
