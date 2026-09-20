"""
Embedder backed by sentence-transformers.

Default model: BAAI/bge-small-en-v1.5, a strong, small, free embedding
model optimized for English retrieval tasks. Loaded lazily and cached
so repeated embedder construction is cheap.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from app.domain.exceptions import EmbeddingError
from app.domain.repositories.embedder import EmbedderPort

logger = logging.getLogger("paperlens.infrastructure.embeddings")


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model '%s'", model_name)
    return SentenceTransformer(model_name)


class SentenceTransformerEmbedder(EmbedderPort):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = _load_model(model_name)
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return [v.tolist() for v in vectors]
        except Exception as exc:
            raise EmbeddingError(f"Document embedding failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            vec = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
            return vec.tolist()
        except Exception as exc:
            raise EmbeddingError(f"Query embedding failed: {exc}") from exc