#checkpoint/json_checkpoint.py
"""
JSON checkpoint of the chunks already indexed (resumable ingestion).

The set is loaded once and kept in memory, so `is_completed` is O(1)
instead of re-reading the file for every chunk. The file is written
atomically, so a crash can never leave a half-written checkpoint.

Only mark a chunk as completed once it is stored EVERYWHERE it is needed
(vector store AND BM25 index, persisted); otherwise a crash leaves the two
indexes out of sync and the missing chunks are never retried.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from app.domain.repositories.checkpoint import CheckpointRepository

logger = logging.getLogger("paperlens.infrastructure.checkpoint")


class JsonCheckpointRepository(CheckpointRepository):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._completed: set[str] = self._read()

    def is_completed(self, chunk_id: str) -> bool:
        return chunk_id in self._completed

    def mark_completed(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return

        self._completed.update(chunk_ids)
        self._write()

    # ------------------------------------------------------------------

    def _read(self) -> set[str]:
        if not self.path.exists():
            return set()

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return set(data.get("completed_chunks", []))

        except (OSError, ValueError, AttributeError) as exc:
            # Re-indexing is safe (upserts are idempotent); crashing is not useful.
            logger.warning(
                "Checkpoint %s is unreadable (%s); starting from scratch",
                self.path,
                exc,
            )
            return set()

    def _write(self) -> None:
        temporary = self.path.with_name(self.path.name + ".tmp")

        temporary.write_text(
            json.dumps({"completed_chunks": sorted(self._completed)}, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)