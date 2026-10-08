
"""
JSON checkpoint for resumable chunk indexing.

The completed chunk IDs are loaded once and kept in memory, making
`is_completed` an O(1) operation without re-reading the file.

Checkpoint writes are atomic: data is written to a temporary file and
then replaced into place, preventing a crash from leaving a partially
written checkpoint.

A chunk must only be marked as completed after it has been successfully
stored in every required index (vector store and BM25) and those indexes
have been persisted.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from app.domain.exceptions import IndexingError
from app.domain.repositories.checkpoint import CheckpointRepository

logger = logging.getLogger("paperlens.infrastructure.checkpoint")

_CHECKPOINT_KEY = "completed_chunks"


class JsonCheckpointRepository(CheckpointRepository):
    """Filesystem-backed checkpoint repository using JSON."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._completed: set[str] = self._read()

    def is_completed(
        self,
        chunk_id: str,
    ) -> bool:
        """Return whether the given chunk has already been completed."""

        return chunk_id in self._completed

    def mark_completed(
        self,
        chunk_ids: list[str],
    ) -> None:
        """
        Mark the given chunk IDs as completed and persist the checkpoint.

        Duplicate IDs are ignored because completed IDs are stored in a set.
        """

        if not chunk_ids:
            return

        self._completed.update(chunk_ids)
        self._write()

    def reset(self) -> None:
        """
        Clear all completed chunk IDs and persist the empty checkpoint.

        This is intentionally exposed through the repository interface so
        callers never need to access implementation details such as
        `_completed` or `_write`.
        """

        self._completed.clear()
        self._write()

        logger.info(
            "Checkpoint reset: %s",
            self.path,
        )

    def count(self) -> int:
        """Return the number of completed chunk IDs."""

        return len(self._completed)

    def _read(self) -> set[str]:
        """Load and validate completed chunk IDs from disk."""

        if not self.path.exists():
            return set()

        try:
            raw_data = self.path.read_text(
                encoding="utf-8",
            )

            data: Any = json.loads(raw_data)

            return self._parse_completed_chunks(data)

        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
        ) as exc:
            logger.warning(
                "Checkpoint %s is unreadable (%s); "
                "starting from scratch",
                self.path,
                exc,
            )

            return set()

    @staticmethod
    def _parse_completed_chunks(
        data: Any,
    ) -> set[str]:
        """
        Validate the checkpoint structure and return completed IDs.

        Invalid structures are treated as an unreadable checkpoint so that
        corrupted data can never cause arbitrary chunks to be considered
        completed.
        """

        if not isinstance(data, dict):
            raise TypeError(
                "checkpoint root must be a JSON object"
            )

        completed = data.get(
            _CHECKPOINT_KEY,
            [],
        )

        if not isinstance(completed, list):
            raise TypeError(
                f"'{_CHECKPOINT_KEY}' must be a list"
            )

        if not all(
            isinstance(chunk_id, str)
            for chunk_id in completed
        ):
            raise TypeError(
                f"all '{_CHECKPOINT_KEY}' values "
                "must be strings"
            )

        return set(completed)

    def _write(self) -> None:
        """Atomically persist the current completed chunk IDs."""

        temporary = self.path.with_name(
            self.path.name + ".tmp"
        )

        try:
            payload = {
                _CHECKPOINT_KEY: sorted(
                    self._completed
                ),
            }

            temporary.write_text(
                json.dumps(
                    payload,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            os.replace(
                temporary,
                self.path,
            )

        except OSError as exc:
            raise IndexingError(
                f"Could not write checkpoint "
                f"'{self.path}': {exc}"
            ) from exc

        finally:
            try:
                if temporary.exists():
                    temporary.unlink()

            except OSError:
                logger.warning(
                    "Could not remove temporary checkpoint "
                    "file %s",
                    temporary,
                )
