from abc import ABC, abstractmethod


class CheckpointRepository(ABC):

    @abstractmethod
    def is_completed(self, chunk_id: str) -> bool:
        ...

    @abstractmethod
    def mark_completed(self, chunk_ids: list[str]) -> None:
        ...