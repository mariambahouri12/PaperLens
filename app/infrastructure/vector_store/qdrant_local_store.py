"""
Qdrant local (on-disk) vector store.

Runs Qdrant in embedded mode via qdrant-client's local path support:
no server, no Docker, no API key. Persists to STORAGE_DIR/vector.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.domain.exceptions import IndexingError
from app.domain.repositories.vector_store import VectorStorePort

logger = logging.getLogger("paperlens.infrastructure.vector_store")

_COLLECTION = "paperlens_chunks"


class QdrantLocalStore(VectorStorePort):
    def __init__(self, path: Path, dimension: int) -> None:
        self.path = path
        self.dimension = dimension
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(path))
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            existing = {c.name for c in self._client.get_collections().collections}
            if _COLLECTION not in existing:
                self._client.create_collection(
                    collection_name=_COLLECTION,
                    vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
                )
                logger.info("Created Qdrant collection '%s' (dim=%d)", _COLLECTION, self.dimension)
        except Exception as exc:
            raise IndexingError(f"Could not initialize Qdrant collection: {exc}") from exc

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        if not ids:
            return
        try:
            points = [
                PointStruct(id=_id_to_int(cid), vector=vec, payload=payload)
                for cid, vec, payload in zip(ids, vectors, payloads)
            ]
            self._client.upsert(collection_name=_COLLECTION, points=points)
            logger.info("Upserted %d vector(s) into Qdrant", len(points))
        except Exception as exc:
            raise IndexingError(f"Qdrant upsert failed: {exc}") from exc

    def search(self, query_vector: list[float], top_k: int):
        try:
            hits = self._client.search(
                collection_name=_COLLECTION,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            logger.warning("Qdrant search failed: %s", exc)
            return []
        return [
            (hit.payload.get("chunk_id", str(hit.id)), float(hit.score), dict(hit.payload))
            for hit in hits
        ]

    def exists(self, chunk_id: str) -> bool:
        try:
            recs = self._client.retrieve(collection_name=_COLLECTION, ids=[_id_to_int(chunk_id)])
            return bool(recs)
        except Exception:
            return False

    def count(self) -> int:
        try:
            return int(self._client.count(collection_name=_COLLECTION).count)
        except Exception:
            return 0


def _id_to_int(chunk_id: str) -> int:
    """Qdrant point IDs must be int or UUID. We hash chunk_id to a
    stable 63-bit int so arbitrary string IDs are supported."""
    import hashlib
    h = hashlib.sha1(chunk_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFFFFFFFFFFFFFF