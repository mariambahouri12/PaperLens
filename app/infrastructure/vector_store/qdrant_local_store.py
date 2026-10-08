# vector_store/qdrant_local_store.py
"""
Qdrant local (on-disk) vector store.

Runs Qdrant in embedded mode via qdrant-client's local path support:
no server, no Docker, no API key. Persists to STORAGE_DIR/vector.

The store holds an OS-level lock on its directory. Use it as a context
manager (or call `close()` explicitly) so the lock is released before the
next run::

    with QdrantLocalStore(path, dimension) as store:
        store.upsert(...)
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from types import TracebackType
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.domain.exceptions import IndexingError
from app.domain.repositories.vector_store import VectorStorePort

logger = logging.getLogger("paperlens.infrastructure.vector_store")

_COLLECTION = "paperlens_chunks"

# Fixed namespace: never change it, or every existing point id changes.
_NAMESPACE = uuid.UUID("6f1c3a52-8e0b-4b7e-9d41-2a5c7e9b0f13")

_UPSERT_BATCH_SIZE = 128


class QdrantLocalStore(VectorStorePort):
    def __init__(self, path: Path, dimension: int) -> None:
        self.path = path
        self.dimension = dimension
        self.path.mkdir(parents=True, exist_ok=True)

        try:
            self._client = QdrantClient(path=str(path))
        except Exception as exc:
            raise IndexingError(
                f"Could not open Qdrant storage at '{path}' "
                f"(is another process using it?): {exc}"
            ) from exc

        self._ensure_collection()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "QdrantLocalStore":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        try:
            if not self._client.collection_exists(_COLLECTION):
                self._client.create_collection(
                    collection_name=_COLLECTION,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    "Created Qdrant collection '%s' (dim=%d)",
                    _COLLECTION,
                    self.dimension,
                )
                return

            existing = self._existing_dimension()

        except Exception as exc:
            raise IndexingError(
                f"Could not initialize Qdrant collection: {exc}"
            ) from exc

        if existing != self.dimension:
            raise IndexingError(
                f"Collection '{_COLLECTION}' already exists with dimension "
                f"{existing}, but the embedder produces {self.dimension}. "
                f"The embedding model or PAPERLENS_EMBEDDING_DIMENSION changed: "
                f"delete '{self.path}' and re-index."
            )

    def _existing_dimension(self) -> int:
        info = self._client.get_collection(_COLLECTION)
        size = getattr(info.config.params.vectors, "size", None)

        if size is None:
            raise IndexingError(
                f"Collection '{_COLLECTION}' has an unexpected vector layout "
                f"(named vectors?)"
            )

        return int(size)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return

        if not (len(ids) == len(vectors) == len(payloads)):
            raise IndexingError(
                f"upsert length mismatch: {len(ids)} ids, "
                f"{len(vectors)} vectors, {len(payloads)} payloads"
            )

        try:
            points = [
                PointStruct(
                    id=_point_id(chunk_id),
                    vector=vector,
                    # chunk_id is always stored so search can return it.
                    payload={**payload, "chunk_id": chunk_id},
                )
                for chunk_id, vector, payload in zip(ids, vectors, payloads)
            ]

            for start in range(0, len(points), _UPSERT_BATCH_SIZE):
                self._client.upsert(
                    collection_name=_COLLECTION,
                    points=points[start : start + _UPSERT_BATCH_SIZE],
                )

            logger.info("Upserted %d vector(s) into Qdrant", len(points))

        except Exception as exc:
            raise IndexingError(f"Qdrant upsert failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(self, query_vector: list[float], top_k: int):
        """Return [(chunk_id, score, payload)] sorted by decreasing score."""
        if top_k <= 0:
            return []

        try:
            response = self._client.query_points(
                collection_name=_COLLECTION,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )

        except Exception as exc:
            # Do not return [] here: an empty result would look like
            # "no relevant chunk" and hide a real storage failure.
            logger.exception("Qdrant search failed")
            raise IndexingError(f"Qdrant search failed: {exc}") from exc

        results = []

        for hit in response.points:
            payload = dict(hit.payload or {})
            results.append(
                (
                    payload.get("chunk_id", str(hit.id)),
                    float(hit.score),
                    payload,
                )
            )

        return results

    def exists(self, chunk_id: str) -> bool:
        try:
            records = self._client.retrieve(
                collection_name=_COLLECTION,
                ids=[_point_id(chunk_id)],
                with_payload=False,
                with_vectors=False,
            )
            return bool(records)

        except Exception as exc:
            logger.warning("Qdrant exists() failed for %s: %s", chunk_id, exc)
            return False

    def count(self) -> int:
        try:
            return int(self._client.count(collection_name=_COLLECTION).count)

        except Exception as exc:
            logger.warning("Qdrant count() failed: %s", exc)
            return 0

    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the storage lock and flush to disk."""
        self._client.close()


def _point_id(chunk_id: str) -> str:
    """
    Qdrant point ids must be an unsigned int or a UUID. A deterministic
    UUID5 of the chunk id supports arbitrary string ids with no realistic
    risk of collision (128 bits), and is stable across runs.
    """
    return str(uuid.uuid5(_NAMESPACE, chunk_id))