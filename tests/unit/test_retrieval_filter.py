from app.application.services.retrieval_filter import filter_chunks


def test_filter_respects_threshold_chunks_and_tokens():
    fused = [
        ("a", 0.05, {}),
        ("b", 0.01, {}),
        ("c", 0.004, {}),   # below threshold
        ("d", 0.02, {}),
    ]
    token_counts = {"a": 100, "b": 100, "c": 100, "d": 100}
    kept = filter_chunks(
        fused,
        token_counts=token_counts,
        min_relevance_score=0.005,
        max_chunks=3,
        max_context_tokens=250,
    )
    ids = [cid for cid, _, _ in kept]
    assert "c" not in ids          # below threshold
    assert len(kept) <= 3          # max_chunks
    assert sum(token_counts[i] for i in ids) <= 250