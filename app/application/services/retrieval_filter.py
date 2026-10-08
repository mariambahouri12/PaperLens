"""
Post-fusion retrieval filtering.

Filtering is applied after RRF and enforces:

1. minimum RRF relevance score;
2. maximum number of chunks;
3. maximum cumulative context token budget.

The fused list is expected to be ordered by decreasing relevance.
"""

from __future__ import annotations

import logging
from typing import Any

from app.domain.exceptions import RetrievalError

logger = logging.getLogger("paperlens.application.filter")

SearchResult = tuple[str, float, dict[str, Any]]


def filter_chunks(
    fused: list[SearchResult],
    token_counts: dict[str, int],
    min_relevance_score: float,
    max_chunks: int,
    max_context_tokens: int,
) -> list[SearchResult]:
    """Apply relevance, count and context-token constraints."""

    if min_relevance_score < 0:
        raise RetrievalError(
            "min_relevance_score cannot be negative"
        )

    if max_chunks <= 0:
        return []

    if max_context_tokens <= 0:
        return []

    selected: list[SearchResult] = []
    total_tokens = 0

    for chunk_id, score, payload in fused:
        if score < min_relevance_score:
            logger.debug(
                "Dropping chunk %s: score %.6f < %.6f",
                chunk_id,
                score,
                min_relevance_score,
            )
            continue

        if len(selected) >= max_chunks:
            break

        if chunk_id not in token_counts:
            raise RetrievalError(
                f"Missing token count for chunk '{chunk_id}'"
            )

        tokens = token_counts[chunk_id]

        if not isinstance(tokens, int):
            raise RetrievalError(
                f"Invalid token count for chunk '{chunk_id}'"
            )

        if tokens < 0:
            raise RetrievalError(
                f"Negative token count for chunk '{chunk_id}'"
            )

        if total_tokens + tokens > max_context_tokens:
            logger.debug(
                "Stopping at chunk %s: token budget exceeded "
                "(%d + %d > %d)",
                chunk_id,
                total_tokens,
                tokens,
                max_context_tokens,
            )
            break

        selected.append(
            (
                chunk_id,
                score,
                payload,
            )
        )

        total_tokens += tokens

    logger.info(
        "Retrieval filtering kept %d chunk(s), total tokens=%d",
        len(selected),
        total_tokens,
    )

    return selected