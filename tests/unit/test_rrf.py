from app.application.services.rrf_fusion import reciprocal_rank_fusion


def test_rrf_merges_and_rewards_shared_results():
    semantic = [("a", 0.9, {"text": "A"}), ("b", 0.8, {"text": "B"})]
    lexical = [("b", 5.0, {"text": "B"}), ("c", 4.0, {"text": "C"})]
    fused = reciprocal_rank_fusion([semantic, lexical], k=60)
    ids = [cid for cid, _, _ in fused]
    assert ids[0] == "b"  # appears in both rankings
    assert set(ids) == {"a", "b", "c"}
    scores = {cid: score for cid, score, _ in fused}
    assert scores["b"] > scores["a"]