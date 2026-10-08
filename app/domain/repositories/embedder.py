# domain/repositories/embedder.py
"""Port: produce dense vector embeddings from text."""
from __future__ import annotations

from abc import ABC, abstractmethod


class EmbedderPort(ABC):
    """
    Dense text embedder.

    `embed_documents` and `embed_query` may use different task prefixes
    (e.g. "search_document: " vs "search_query: " for Nomic models), so
    they are separate methods rather than one generic `embed`.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output vector size. Must match the vector store's dimension."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunks for indexing. Order is preserved.
        Raises EmbeddingError on failure."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query. Raises EmbeddingError on failure."""