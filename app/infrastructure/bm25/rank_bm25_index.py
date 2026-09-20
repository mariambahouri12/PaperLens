"""
BM25 index backed by rank_bm25 (pure Python, MIT).

The index is held in memory and persisted to disk as a pickle so it can
be reloaded on subsequent runs without re-ingesting documents.
"""
from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.domain.exceptions import IndexingError
from app.domain.repositories.bm25_index import BM25IndexPort

logger = logging.getLogger("paperlens.infrastructure.bm25")

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class RankBM25Index(BM25IndexPort):
    def __init__(self, persist_path: Path) -> None:
        self.persist_path = persist_path
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self._index_file = persist_path / "bm25.pkl"

        self._ids: list[str] = []
        self._payloads: dict[str, dict[str, Any]] = {}
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def add(self, chunk_ids: list[str], texts: list[str], payloads: list[dict[str, Any]]) -> None:
        for cid, text, payload in zip(chunk_ids, texts, payloads):
            if cid in self._payloads:
                continue
            self._ids.append(cid)
            self._tokenized.append(_tokenize(text))
            self._payloads[cid] = payload
        try:
            self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
            logger.info("BM25 index now contains %d document(s)", len(self._ids))
        except Exception as exc:
            raise IndexingError(f"BM25 rebuild failed: {exc}") from exc

    def search(self, query: str, top_k: int):
        if self._bm25 is None or not self._ids:
            return []
        try:
            scores = self._bm25.get_scores(_tokenize(query))
        except Exception as exc:
            logger.warning("BM25 search failed: %s", exc)
            return []
        ranked = sorted(zip(self._ids, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(cid, float(score), self._payloads[cid]) for cid, score in ranked]

    def persist(self) -> None:
        try:
            with open(self._index_file, "wb") as fh:
                pickle.dump({
                    "ids": self._ids,
                    "tokenized": self._tokenized,
                    "payloads": self._payloads,
                }, fh)
            logger.info("BM25 index persisted to %s", self._index_file)
        except Exception as exc:
            logger.warning("Could not persist BM25 index: %s", exc)

    def load(self) -> None:
        if not self._index_file.exists():
            return
        try:
            with open(self._index_file, "rb") as fh:
                data = pickle.load(fh)
            self._ids = data["ids"]
            self._tokenized = data["tokenized"]
            self._payloads = data["payloads"]
            self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
            logger.info("BM25 index loaded: %d document(s)", len(self._ids))
        except Exception as exc:
            logger.warning("Could not load BM25 index: %s", exc)

    def count(self) -> int:
        return len(self._ids)