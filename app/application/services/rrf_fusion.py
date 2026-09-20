"""
Reciprocal Rank Fusion.

Combines ranked lists (here: semantic and BM25 results) into a single
ranking. The RRF score of a document d is:

    score(d) = Σ_r  1 / (k + rank_r(d))

where r iterates over the input rankings and rank_r(d) is the rank of
d in ranking r (starting at 1). `k` is the RRF_K parameter from
settings; a value of 60 is the widely-used default.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("paperlens.application.rrf")


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float, dict[str, Any]]]],
    k: int = 60,
) -> list[tuple[str, float, dict[str, Any]]]:
    """
    Parameters
    ----------
    rankings : list of ranked lists. Each ranked list is a list of
               (chunk_id, score, payload) sorted by score descending.
    k        : RRF smoothing constant.

    Returns
    -------
    A single ranked list of (chunk_id, rrf_score, payload), sorted by
    rrf_score descending.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}

    for ranking in rankings:
        for rank, (chunk_id, _score, payload) in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            payloads.setdefault(chunk_id, payload)

    fused = [(cid, scores[cid], payloads[cid]) for cid in scores]
    fused.sort(key=lambda x: x[1], reverse=True)
    logger.debug("RRF fused %d rankings into %d unique chunks", len(rankings), len(fused))
    return fused