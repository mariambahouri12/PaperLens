"""
Qdrant local (on-disk) vector store.

Runs Qdrant in embedded mode via qdrant-client's local path support:
no server, no Docker, no API key. Persists to the configured storage path.

The store holds an OS-level lock on its directory. Use it as a context
manager, or call `close()` explicitly, so the lock is released before
the next run.

Example:
    with QdrantLocalStore(path, dimension) as store:
        store.upsert(ids, vectors, payloads)
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
from app.domain.repositories.vector_store import (
    SearchResult,
    VectorStorePort,
)

logger = logging.getLogger("paperlens.infrastructure.vector_store")

_COLLECTION = "paperlens_chunks"

# Fixed namespace: changing it would change every existing point ID.
_NAMESPACE = uuid.UUID("6f1c3a52-8e0b-4b7e-9d41-2a5c7e9b0f13")

_UPSERT_BATCH_SIZE = 128


class QdrantLocalStore(VectorStorePort):
    """Qdrant-backed implementation of the vector store port."""

    def __init__(self, path: Path, dimension: int) -> None:
        if dimension <= 0:
            raise IndexingError(
                f"Embedding dimension must be positive, got {dimension}"
            )

        self.path = path
        self.dimension = dimension
        self._closed = False

        self.path.mkdir(parents=True, exist_ok=True)

        try:
            self._client = QdrantClient(path=str(self.path))
            self._ensure_collection()
        except IndexingError:
            self._safe_close_client()
            raise
        except Exception as exc:
            self._safe_close_client()
            raise IndexingError(
                f"Could not initialize Qdrant storage at "
                f"'{self.path}': {exc}"
            ) from exc

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

            existing_dimension = self._existing_dimension()

        except IndexingError:
            raise
        except Exception as exc:
            raise IndexingError(
                f"Could not initialize Qdrant collection: {exc}"
            ) from exc

        if existing_dimension != self.dimension:
            raise IndexingError(
                f"Collection '{_COLLECTION}' already exists with "
                f"dimension {existing_dimension}, but the embedder "
                f"produces {self.dimension}. The embedding model or "
                f"PAPERLENS_EMBEDDING_DIMENSION changed: delete "
                f"'{self.path}' and re-index."
            )

    def _existing_dimension(self) -> int:
        """Return the configured vector dimension of the collection."""
        info = self._client.get_collection(_COLLECTION)
        vectors = info.config.params.vectors
        size = getattr(vectors, "size", None)

        if size is None:
            raise IndexingError(
                f"Collection '{_COLLECTION}' has an unexpected vector "
                f"layout (named vectors?)"
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
        """Insert or replace vectors in bounded batches."""
        if not ids:
            return

        if not (len(ids) == len(vectors) == len(payloads)):
            raise IndexingError(
                f"upsert length mismatch: {len(ids)} ids, "
                f"{len(vectors)} vectors, {len(payloads)} payloads"
            )

        self._ensure_open()

        try:
            points = [
                PointStruct(
                    id=_point_id(chunk_id),
                    vector=vector,
                    payload={
                        **payload,
                        "chunk_id": chunk_id,
                    },
                )
                for chunk_id, vector, payload in zip(
                    ids,
                    vectors,
                    payloads,
                )
            ]

            for start in range(0, len(points), _UPSERT_BATCH_SIZE):
                batch = points[start : start + _UPSERT_BATCH_SIZE]

                self._client.upsert(
                    collection_name=_COLLECTION,
                    points=batch,
                )

            logger.info(
                "Upserted %d vector(s) into Qdrant",
                len(points),
            )

        except IndexingError:
            raise
        except Exception as exc:
            raise IndexingError(
                f"Qdrant upsert failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        top_k: int,
    ) -> list[SearchResult]:
        """Return search results sorted by decreasing similarity score."""
        if top_k <= 0:
            return []

        self._ensure_open()

        try:
            response = self._client.query_points(
                collection_name=_COLLECTION,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )

        except Exception as exc:
            logger.exception("Qdrant search failed")
            raise IndexingError(
                f"Qdrant search failed: {exc}"
            ) from exc

        results: list[SearchResult] = []

        for hit in response.points:
            payload = dict(hit.payload or {})
            chunk_id = payload.get("chunk_id", str(hit.id))

            results.append(
                (
                    str(chunk_id),
                    float(hit.score),
                    payload,
                )
            )

        return results

    def exists(self, chunk_id: str) -> bool:
        """Return whether the chunk has a stored Qdrant point."""
        self._ensure_open()

        try:
            records = self._client.retrieve(
                collection_name=_COLLECTION,
                ids=[_point_id(chunk_id)],
                with_payload=False,
                with_vectors=False,
            )

            return bool(records)

        except Exception as exc:
            raise IndexingError(
                f"Qdrant exists() failed for '{chunk_id}': {exc}"
            ) from exc

    def count(self) -> int:
        """Return the number of stored points."""
        self._ensure_open()

        try:
            return int(
                self._client.count(
                    collection_name=_COLLECTION,
                ).count
            )

        except Exception as exc:
            raise IndexingError(
                f"Qdrant count() failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the Qdrant storage lock. Safe to call multiple times."""
        if self._closed:
            return

        try:
            self._client.close()
        except Exception as exc:
            raise IndexingError(
                f"Could not close Qdrant storage at "
                f"'{self.path}': {exc}"
            ) from exc
        finally:
            self._closed = True

    def _ensure_open(self) -> None:
        """Reject operations after the store has been closed."""
        if self._closed:
            raise IndexingError("Qdrant store is closed")

    def _safe_close_client(self) -> None:
        """Best-effort cleanup when initialization fails."""
        client = getattr(self, "_client", None)

        if client is None:
            return

        try:
            client.close()
        except Exception:
            logger.warning(
                "Could not close Qdrant client after initialization failure",
                exc_info=True,
            )


def _point_id(chunk_id: str) -> str:
    """
    Convert an arbitrary chunk ID into a deterministic Qdrant UUID.

    Qdrant point IDs support unsigned integers or UUIDs. UUID5 provides
    stable IDs across runs while supporting arbitrary string chunk IDs.
    """
    return str(uuid.uuid5(_NAMESPACE, chunk_id))
