"""
Hybrid retrieval service.

Combines dense semantic retrieval and BM25 lexical retrieval, then
fuses both rankings using Reciprocal Rank Fusion (RRF).

This service belongs to the application layer and depends only on
domain ports and application-level RRF logic.
"""

from __future__ import annotations

import logging
from typing import Any

from app.application.services.rrf_fusion import reciprocal_rank_fusion
from app.domain.repositories.bm25_index import BM25IndexPort
from app.domain.repositories.embedder import EmbedderPort
from app.domain.repositories.vector_store import VectorStorePort

logger = logging.getLogger("paperlens.application.retrieval")

SearchResult = tuple[str, float, dict[str, Any]]


class HybridRetriever:
    """Retrieve documents using dense + lexical search with RRF fusion."""

    def __init__(
        self,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
        bm25_index: BM25IndexPort,
        rrf_k: int,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")

        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        """Run dense + BM25 retrieval and fuse both rankings."""
        if not query.strip():
            return []

        if top_k <= 0:
            return []

        semantic = self._semantic_search(query, top_k)
        lexical = self._bm25_search(query, top_k)

        if not semantic and not lexical:
            return []

        fused = reciprocal_rank_fusion(
            [semantic, lexical],
            k=self.rrf_k,
        )

        logger.info(
            "Hybrid retrieval: semantic=%d, bm25=%d, fused=%d",
            len(semantic),
            len(lexical),
            len(fused),
        )

        return fused

    def _semantic_search(
        self,
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        try:
            query_vector = self.embedder.embed_query(query)
            return self.vector_store.search(
                query_vector,
                top_k=top_k,
            )
        except Exception:
            logger.exception("Semantic retrieval failed")
            return []

    def _bm25_search(
        self,
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        try:
            return self.bm25_index.search(
                query,
                top_k=top_k,
            )
        except Exception:
            logger.exception("BM25 retrieval failed")
            return []