"""
Post-fusion filtering.

Applied AFTER RRF. Enforces three independent constraints:

  1. MIN_RELEVANCE_SCORE : drop chunks whose RRF score is below the
                           threshold. Note that RRF scores are tiny
                           (typically in the 0.01–0.05 range for
                           rank-1 chunks with k=60), so the default
                           threshold in settings is calibrated for RRF.
  2. MAX_CHUNKS          : cap the number of returned chunks.
  3. MAX_CONTEXT_TOKENS  : stop adding chunks when the cumulative
                           token count would exceed the budget.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("paperlens.application.filter")


def filter_chunks(
    fused: list[tuple[str, float, dict[str, Any]]],
    token_counts: dict[str, int],
    min_relevance_score: float,
    max_chunks: int,
    max_context_tokens: int,
) -> list[tuple[str, float, dict[str, Any]]]:
    selected: list[tuple[str, float, dict[str, Any]]] = []
    total_tokens = 0

    for chunk_id, score, payload in fused:
        if score < min_relevance_score:
            logger.debug("Drop %s: score %.6f < %.6f", chunk_id, score, min_relevance_score)
            continue
        if len(selected) >= max_chunks:
            break
        tokens = token_counts.get(chunk_id, 0)
        if total_tokens + tokens > max_context_tokens:
            logger.debug("Stop adding %s: would exceed token budget (%d + %d > %d)",
                         chunk_id, total_tokens, tokens, max_context_tokens)
            break
        selected.append((chunk_id, score, payload))
        total_tokens += tokens

    logger.info("Filtering kept %d chunk(s), total tokens=%d", len(selected), total_tokens)
    return selected