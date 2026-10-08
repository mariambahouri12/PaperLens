
"""
BM25 index backed by rank_bm25 (pure Python, MIT).

The index is held in memory and persisted to disk as a pickle so it can
be reloaded on subsequent runs without re-ingesting documents.

The BM25 model is rebuilt lazily: `add` only stores tokenized documents,
and the model is rebuilt once, on the next search, regardless of how
many times `add` was called.
"""

from __future__ import annotations

import heapq
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.domain.exceptions import IndexingError
from app.domain.repositories.bm25_index import BM25IndexPort, SearchResult


logger = logging.getLogger("paperlens.infrastructure.bm25")

_TOKEN_RE = re.compile(r"\w+")
_INDEX_FILENAME = "bm25.pkl"


def _tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return [token.lower() for token in _TOKEN_RE.findall(text)]


class RankBM25Index(BM25IndexPort):
    """In-memory BM25 index with atomic persistence."""

    def __init__(self, persist_path: Path) -> None:
        self.persist_path = persist_path
        self.persist_path.mkdir(parents=True, exist_ok=True)

        self._index_file = self.persist_path / _INDEX_FILENAME

        self._ids: list[str] = []
        self._payloads: dict[str, dict[str, Any]] = {}
        self._tokenized: list[list[str]] = []

        self._bm25: BM25Okapi | None = None
        self._model_stale = False

    def add(
        self,
        ids: list[str],
        texts: list[str],
        payloads: list[dict[str, Any]],
    ) -> None:
        """
        Add documents to the index.

        Existing chunk IDs are ignored, making ingestion idempotent.
        """
        if not (
            len(ids) == len(texts) == len(payloads)
        ):
            raise IndexingError(
                "ids, texts and payloads must have the same length"
            )

        if not ids:
            return

        for chunk_id, text, payload in zip(
            ids,
            texts,
            payloads,
            strict=True,
        ):
            if not isinstance(chunk_id, str) or not chunk_id:
                raise IndexingError(
                    "Every chunk ID must be a non-empty string"
                )

            if not isinstance(text, str):
                raise IndexingError(
                    f"Text for chunk '{chunk_id}' must be a string"
                )

            if not isinstance(payload, dict):
                raise IndexingError(
                    f"Payload for chunk '{chunk_id}' must be a dictionary"
                )

            if chunk_id in self._payloads:
                continue

            self._ids.append(chunk_id)
            self._tokenized.append(_tokenize(text))
            self._payloads[chunk_id] = dict(payload)

        self._model_stale = True

    def persist(self) -> None:
        """
        Persist the index atomically.

        Data is first written to a temporary file and then atomically
        replaced into the final location.
        """
        temporary = self._index_file.with_suffix(".pkl.tmp")

        payload = {
            "ids": self._ids,
            "tokenized": self._tokenized,
            "payloads": self._payloads,
        }

        try:
            with temporary.open("wb") as file:
                pickle.dump(
                    payload,
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            os.replace(temporary, self._index_file)

            logger.info(
                "Persisted BM25 index with %d documents to %s",
                len(self._ids),
                self._index_file,
            )

        except Exception as exc:
            raise IndexingError(
                f"Could not persist BM25 index to "
                f"'{self._index_file}': {exc}"
            ) from exc

        finally:
            try:
                if temporary.exists():
                    temporary.unlink()
            except OSError:
                logger.warning(
                    "Could not remove temporary BM25 file %s",
                    temporary,
                )

    def load(self) -> None:
        """
        Load a previously persisted index.

        Missing index files are treated as a no-op.
        Invalid or corrupted files raise IndexingError.
        """
        if not self._index_file.exists():
            return

        try:
            with self._index_file.open("rb") as file:
                data = pickle.load(file)

            ids, tokenized, payloads = self._validate_loaded_data(data)

            self._ids = ids
            self._tokenized = tokenized
            self._payloads = payloads

            self._bm25 = None
            self._model_stale = True

            logger.info(
                "Loaded BM25 index with %d documents from %s",
                len(self._ids),
                self._index_file,
            )

        except IndexingError:
            raise

        except Exception as exc:
            raise IndexingError(
                f"Could not load BM25 index from "
                f"'{self._index_file}': {exc}"
            ) from exc

    @staticmethod
    def _validate_loaded_data(
        data: Any,
    ) -> tuple[
        list[str],
        list[list[str]],
        dict[str, dict[str, Any]],
    ]:
        """Validate the structure of persisted BM25 data."""

        if not isinstance(data, dict):
            raise IndexingError(
                "Persisted BM25 data must be a dictionary"
            )

        ids = data.get("ids")
        tokenized = data.get("tokenized")
        payloads = data.get("payloads")

        if not isinstance(ids, list):
            raise IndexingError(
                "Persisted BM25 'ids' must be a list"
            )

        if not isinstance(tokenized, list):
            raise IndexingError(
                "Persisted BM25 'tokenized' must be a list"
            )

        if not isinstance(payloads, dict):
            raise IndexingError(
                "Persisted BM25 'payloads' must be a dictionary"
            )

        if len(ids) != len(tokenized):
            raise IndexingError(
                "Persisted BM25 ids and tokenized data have "
                "inconsistent lengths"
            )

        if not all(
            isinstance(chunk_id, str) and chunk_id
            for chunk_id in ids
        ):
            raise IndexingError(
                "Persisted BM25 IDs must be non-empty strings"
            )

        if len(set(ids)) != len(ids):
            raise IndexingError(
                "Persisted BM25 IDs must be unique"
            )

        if not all(isinstance(tokens, list) for tokens in tokenized):
            raise IndexingError(
                "Persisted BM25 tokenized data must contain lists"
            )

        if not all(
            isinstance(token, str)
            for tokens in tokenized
            for token in tokens
        ):
            raise IndexingError(
                "Persisted BM25 tokens must be strings"
            )

        if set(ids) != set(payloads):
            raise IndexingError(
                "Persisted BM25 IDs and payload keys are inconsistent"
            )

        if not all(
            isinstance(payload, dict)
            for payload in payloads.values()
        ):
            raise IndexingError(
                "Persisted BM25 payload values must be dictionaries"
            )

        return ids, tokenized, payloads

    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        """
        Search the BM25 index.

        Only documents with strictly positive BM25 scores are returned.
        Results are deterministically ordered by descending score and
        then ascending chunk ID.
        """
        if top_k <= 0:
            return []

        if not self._ids:
            return []

        query_tokens = _tokenize(query)

        if not query_tokens:
            return []

        try:
            model = self._model()
            scores = model.get_scores(query_tokens)

        except Exception as exc:
            raise IndexingError(
                f"BM25 search failed: {exc}"
            ) from exc

        candidates = [
            (chunk_id, float(score))
            for chunk_id, score in zip(
                self._ids,
                scores,
                strict=True,
            )
            if float(score) > 0
        ]

        best = heapq.nsmallest(
            top_k,
            candidates,
            key=lambda item: (-item[1], item[0]),
        )

        return [
            (
                chunk_id,
                score,
                dict(self._payloads[chunk_id]),
            )
            for chunk_id, score in best
        ]

    def count(self) -> int:
        """Return the number of indexed documents."""
        return len(self._ids)

    def _model(self) -> BM25Okapi:
        """Return the current BM25 model, rebuilding it when necessary."""
        if self._bm25 is None or self._model_stale:
            self._bm25 = BM25Okapi(self._tokenized)
            self._model_stale = False

        return self._bm25
