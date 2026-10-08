"""
Reciprocal Rank Fusion.

Combines multiple ranked retrieval lists into a single ranking.

For each document d:

    RRF(d) = Σ 1 / (k + rank(d))

Ranks start at 1.

The input rankings are expected to already be ordered by their
respective retrieval systems.
"""

from __future__ import annotations

import logging
from typing import Any

from app.domain.exceptions import RetrievalError

logger = logging.getLogger("paperlens.application.rrf")

SearchResult = tuple[str, float, dict[str, Any]]


def reciprocal_rank_fusion(
    rankings: list[list[SearchResult]],
    k: int = 60,
) -> list[SearchResult]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    The first payload encountered for a chunk is preserved.
    """

    if k <= 0:
        raise RetrievalError(
            "RRF parameter k must be greater than zero"
        )

    if not rankings:
        return []

    scores: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}

    for ranking in rankings:
        seen_in_ranking: set[str] = set()

        for rank, result in enumerate(ranking, start=1):
            chunk_id, _score, payload = result

            if not isinstance(chunk_id, str) or not chunk_id:
                raise RetrievalError(
                    "RRF received an invalid chunk ID"
                )

            if not isinstance(payload, dict):
                raise RetrievalError(
                    f"RRF received an invalid payload "
                    f"for chunk '{chunk_id}'"
                )

            # A valid ranking must contain each document at most once.
            if chunk_id in seen_in_ranking:
                logger.warning(
                    "Duplicate chunk '%s' found in one ranking; "
                    "ignoring the duplicate",
                    chunk_id,
                )
                continue

            seen_in_ranking.add(chunk_id)

            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + 1.0 / (k + rank)
            )

            # Semantic and BM25 payloads should describe the same chunk.
            # Keep the first payload deterministically.
            payloads.setdefault(
                chunk_id,
                dict(payload),
            )

    fused = [
        (
            chunk_id,
            scores[chunk_id],
            payloads[chunk_id],
        )
        for chunk_id in scores
    ]

    # Deterministic ordering:
    #   1. highest RRF score
    #   2. chunk ID as tie-breaker
    fused.sort(
        key=lambda item: (-item[1], item[0])
    )

    logger.debug(
        "RRF fused %d ranking(s) into %d unique chunk(s)",
        len(rankings),
        len(fused),
    )

    return fused