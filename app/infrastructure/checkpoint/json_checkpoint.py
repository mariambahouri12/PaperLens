import json
from pathlib import Path

from app.domain.repositories.checkpoint import CheckpointRepository


class JsonCheckpointRepository(CheckpointRepository):

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> set[str]:
        if not self.path.exists():
            return set()

        data = json.loads(self.path.read_text(encoding="utf-8"))
        return set(data.get("completed_chunks", []))

    def is_completed(self, chunk_id: str) -> bool:
        return chunk_id in self._load()

    def mark_completed(self, chunk_ids: list[str]) -> None:
        completed = self._load()
        completed.update(chunk_ids)

        self.path.write_text(
            json.dumps(
                {"completed_chunks": sorted(completed)},
                indent=2,
            ),
            encoding="utf-8",
        )