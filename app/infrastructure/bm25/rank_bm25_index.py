# bm25/rank_bm25_index.py
"""
BM25 index backed by rank_bm25 (pure Python, MIT).

The index is held in memory and persisted to disk as a pickle so it can
be reloaded on subsequent runs without re-ingesting documents.

The BM25 model is rebuilt lazily: `add` only stores tokens, and the model
is rebuilt once, on the next search, however many times `add` was called.
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
from app.domain.repositories.bm25_index import BM25IndexPort

logger = logging.getLogger("paperlens.infrastructure.bm25")

# \w keeps accented letters, Greek letters, digits and underscores.
_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


class RankBM25Index(BM25IndexPort):
    def __init__(self, persist_path: Path) -> None:
        self.persist_path = persist_path
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self._index_file = persist_path / "bm25.pkl"

        self._ids: list[str] = []
        self._payloads: dict[str, dict[str, Any]] = {}
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._model_stale = False

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        chunk_ids: list[str],
        texts: list[str],
        payloads: list[dict[str, Any]],
    ) -> None:
        if not (len(chunk_ids) == len(texts) == len(payloads)):
            raise IndexingError(
                f"BM25 add length mismatch: {len(chunk_ids)} ids, "
                f"{len(texts)} texts, {len(payloads)} payloads"
            )

        added = 0

        for chunk_id, text, payload in zip(chunk_ids, texts, payloads):
            if chunk_id in self._payloads:
                continue

            self._ids.append(chunk_id)
            self._tokenized.append(_tokenize(text))
            self._payloads[chunk_id] = payload
            added += 1

        if added:
            self._model_stale = True
            logger.info("BM25 index now contains %d document(s)", len(self._ids))

    def persist(self) -> None:
        temporary = self._index_file.with_suffix(".pkl.tmp")

        try:
            with open(temporary, "wb") as handle:
                pickle.dump(
                    {
                        "ids": self._ids,
                        "tokenized": self._tokenized,
                        "payloads": self._payloads,
                    },
                    handle,
                )

            os.replace(temporary, self._index_file)
            logger.info("BM25 index persisted to %s", self._index_file)

        except Exception as exc:
            # Must not be silent: the checkpoint would mark chunks as done
            # while the BM25 index on disk does not contain them.
            raise IndexingError(f"Could not persist BM25 index: {exc}") from exc

    def load(self) -> None:
        if not self._index_file.exists():
            return

        try:
            with open(self._index_file, "rb") as handle:
                data = pickle.load(handle)

            self._ids = data["ids"]
            self._tokenized = data["tokenized"]
            self._payloads = data["payloads"]
            self._model_stale = True  # model rebuilt on the next search

            logger.info("BM25 index loaded: %d document(s)", len(self._ids))

        except Exception as exc:
            raise IndexingError(
                f"BM25 index file '{self._index_file}' is unreadable ({exc}). "
                f"Delete it and re-ingest the documents."
            ) from exc

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int):
        """Return [(chunk_id, score, payload)]; only chunks sharing a term."""
        if top_k <= 0 or not self._ids:
            return []

        tokens = _tokenize(query)

        if not tokens:
            return []

        try:
            scores = self._model().get_scores(tokens)

        except Exception as exc:
            logger.exception("BM25 search failed")
            raise IndexingError(f"BM25 search failed: {exc}") from exc

        # score > 0 only: a chunk sharing no term with the query must not
        # enter the fusion with an arbitrary rank. Ties are broken by id,
        # so results are deterministic.
        best = heapq.nlargest(
            top_k,
            (
                (float(score), chunk_id)
                for chunk_id, score in zip(self._ids, scores)
                if score > 0
            ),
        )

        return [
            (chunk_id, score, self._payloads[chunk_id])
            for score, chunk_id in best
        ]

    def count(self) -> int:
        return len(self._ids)

    # ------------------------------------------------------------------

    def _model(self) -> BM25Okapi:
        if self._bm25 is None or self._model_stale:
            self._bm25 = BM25Okapi(self._tokenized)
            self._model_stale = False

        return self._bm25