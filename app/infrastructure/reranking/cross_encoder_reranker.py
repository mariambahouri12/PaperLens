"""Cross-Encoder implementation of the reranker port."""
from __future__ import annotations

import logging
from typing import Any

from sentence_transformers import CrossEncoder

from app.domain.repositories.reranker import (
    RerankerPort,
    SearchResult,
)

logger = logging.getLogger("paperlens.infrastructure.reranking")


class CrossEncoderReranker(RerankerPort):
    """Rerank retrieved chunks with a Cross-Encoder model."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
    ) -> None:
        self.model = CrossEncoder(
            model_name,
            device=device,
        )
        logger.info(
            "Cross-Encoder loaded: model=%s, device=%s",
            model_name,
            device,
        )

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        if not query.strip() or top_k <= 0 or not candidates:
            return []

        valid_indices: list[int] = []
        pairs: list[tuple[str, str]] = []

        for index, (_, _, payload) in enumerate(candidates):
            text = payload.get("text")

            if isinstance(text, str) and text.strip():
                valid_indices.append(index)
                pairs.append((query, text))

        if not pairs:
            logger.warning(
                "No candidates contain usable text for reranking."
            )
            return candidates[:top_k]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        ranked_indices = [
            index
            for index, _ in sorted(
                zip(valid_indices, scores),
                key=lambda item: float(item[1]),
                reverse=True,
            )
        ]

        # Candidates without text remain at the end in their
        # original order.
        ranked_indices.extend(
            index
            for index in range(len(candidates))
            if index not in set(valid_indices)
        )

        # Preserve the original RRF score. The Cross-Encoder score
        # determines ordering only; it is not a retrieval threshold.
        return [
            candidates[index]
            for index in ranked_indices[:top_k]
        ]