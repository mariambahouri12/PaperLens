"""Configurable retrieval with optional Cross-Encoder reranking."""
from __future__ import annotations

import logging
from typing import Any

from app.application.services.rrf_fusion import reciprocal_rank_fusion
from app.domain.repositories.bm25_index import BM25IndexPort
from app.domain.repositories.embedder import EmbedderPort
from app.domain.repositories.reranker import RerankerPort
from app.domain.repositories.vector_store import VectorStorePort

logger = logging.getLogger("paperlens.application.retrieval")

SearchResult = tuple[str, float, dict[str, Any]]


class HybridRetriever:
    """Retrieve, fuse and optionally rerank document chunks."""

    def __init__(
        self,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
        bm25_index: BM25IndexPort,
        rrf_k: int,
        use_bm25: bool,
        use_dense: bool,
        use_rrf: bool,
        reranker: RerankerPort | None = None,
        use_reranker: bool = False,
        reranker_candidate_k: int = 30,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")

        if not use_bm25 and not use_dense:
            raise ValueError(
                "At least one retrieval method must be enabled: "
                "use_bm25 or use_dense."
            )

        if use_rrf and not (use_bm25 and use_dense):
            raise ValueError(
                "RRF requires both BM25 and Dense retrieval to be enabled."
            )

        if use_bm25 and use_dense and not use_rrf:
            raise ValueError(
                "BM25 + Dense without RRF is not supported yet."
            )

        if reranker_candidate_k <= 0:
            raise ValueError(
                "reranker_candidate_k must be greater than zero"
            )

        if use_reranker and not use_rrf:
            raise ValueError(
                "Cross-Encoder reranking requires hybrid retrieval with RRF."
            )

        if use_reranker and reranker is None:
            raise ValueError(
                "A reranker must be provided when use_reranker=True."
            )

        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.rrf_k = rrf_k

        self.use_bm25 = use_bm25
        self.use_dense = use_dense
        self.use_rrf = use_rrf

        self.reranker = reranker
        self.use_reranker = use_reranker
        self.reranker_candidate_k = reranker_candidate_k

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        """Retrieve candidates, fuse them and optionally rerank."""
        if not query.strip() or top_k <= 0:
            return []

        search_k = (
            max(top_k, self.reranker_candidate_k)
            if self.use_reranker
            else top_k
        )

        semantic: list[SearchResult] = []
        lexical: list[SearchResult] = []

        if self.use_dense:
            semantic = self._semantic_search(query, search_k)

        if self.use_bm25:
            lexical = self._bm25_search(query, search_k)

        if self.use_rrf:
            if not semantic and not lexical:
                return []

            fused = reciprocal_rank_fusion(
                [semantic, lexical],
                k=self.rrf_k,
            )

            logger.info(
                "Hybrid retrieval: dense=%d, bm25=%d, rrf=%d",
                len(semantic),
                len(lexical),
                len(fused),
            )

            if not self.use_reranker:
                return fused[:top_k]

            # Restrict reranking to the best RRF candidates.
            candidates = fused[:self.reranker_candidate_k]

            try:
                assert self.reranker is not None
                reranked = self.reranker.rerank(
                    query=query,
                    candidates=candidates,
                    top_k=top_k,
                )

                logger.info(
                    "Cross-Encoder reranking: candidates=%d, results=%d",
                    len(candidates),
                    len(reranked),
                )
                return reranked[:top_k]

            except Exception:
                logger.exception(
                    "Cross-Encoder reranking failed; falling back to RRF."
                )
                return candidates[:top_k]

        results = semantic if self.use_dense else lexical

        logger.info(
            "Retrieval mode=%s, results=%d",
            "dense" if self.use_dense else "bm25",
            len(results),
        )

        return results[:top_k]

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