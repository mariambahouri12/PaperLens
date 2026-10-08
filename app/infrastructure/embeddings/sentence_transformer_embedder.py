# infrastructure/embeddings/sentence_transformer_embedder.py
"""
Embedder backed by sentence-transformers.

Default model: nomic-ai/nomic-embed-text-v1.5 (768 dimensions, Matryoshka,
8192-token context). Everything that depends on the model (task prefixes,
output dimension, max sequence length, device) is read from settings, so
switching model is a configuration change, not a code change.

The underlying `SentenceTransformer` instance is loaded on first
instantiation and cached by configuration, so building several
embedders with the same parameters is cheap. Two embedders built
with different settings (model, device, dimension, ...) get two
different cached models.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from app.config.settings import settings
from app.domain.exceptions import EmbeddingError
from app.domain.repositories.embedder import EmbedderPort

logger = logging.getLogger("paperlens.infrastructure.embeddings")


@lru_cache(maxsize=4)
def _load_model(
    model_name: str,
    device: str | None,
    trust_remote_code: bool,
    dimension: int,
    max_seq_length: int,
):
    """
    Return a cached `SentenceTransformer` for the given configuration.

    The cache key contains every parameter that influences the loaded
    model, so two callers with different settings never share the wrong
    instance.

    `sentence_transformers` is imported here (not at module level) so
    this module can be imported in environments where the library is
    not installed (e.g. linting, unit tests that mock the model).
    """
    from sentence_transformers import SentenceTransformer

    logger.info(
        "Loading embedding model '%s' (device=%s, dimension=%d)",
        model_name,
        device or "auto",
        dimension,
    )

    model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=trust_remote_code,
        truncate_dim=dimension,
    )
    model.max_seq_length = max_seq_length

    return model


class SentenceTransformerEmbedder(EmbedderPort):
    """
    `EmbedderPort` implementation backed by sentence-transformers.

    See the module docstring for the model and configuration notes.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model

        try:
            self._model = _load_model(
                self.model_name,
                settings.embedding_device,
                settings.embedding_trust_remote_code,
                settings.embedding_dimension,
                settings.embedding_max_seq_length,
            )
            self._dimension = self._probe_dimension()

        except EmbeddingError:
            raise

        except Exception as exc:
            # Any SDK error (download failure, invalid model, ...) is
            # translated to EmbeddingError so callers only handle one type.
            raise EmbeddingError(
                f"Could not load embedding model '{self.model_name}': {exc}"
            ) from exc

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            return self._encode(texts, settings.embedding_document_prefix)

        except Exception as exc:
            raise EmbeddingError(
                f"Document embedding failed: {exc}"
            ) from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._encode([text], settings.embedding_query_prefix)[0]

        except Exception as exc:
            raise EmbeddingError(
                f"Query embedding failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        vectors = self._model.encode(
            [prefix + text for text in texts],
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=settings.embedding_normalize,
            show_progress_bar=False,
        )

        # numpy array -> list[list[float]]: JSON-serializable, ready for
        # any vector store.
        return [vector.tolist() for vector in vectors]

    def _probe_dimension(self) -> int:
        """
        Measure the real output size and make sure it matches the
        settings, so the vector store is never created with the wrong
        dimension.

        Uses a real `encode` call rather than
        `get_sentence_embedding_dimension()` because the latter may
        return the pre-truncation size for models that support
        Matryoshka truncation.
        """
        vectors = self._model.encode(
            ["dimension probe"],
            normalize_embeddings=settings.embedding_normalize,
            show_progress_bar=False,
        )
        size = int(vectors.shape[1])

        if size != settings.embedding_dimension:
            raise EmbeddingError(
                f"Model '{self.model_name}' produced {size}-dimensional "
                f"vectors but PAPERLENS_EMBEDDING_DIMENSION is "
                f"{settings.embedding_dimension}"
            )

        return size