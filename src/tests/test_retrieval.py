"""Integration tests for the retrieval ranking logic.

These tests verify the ranking behavior of `hybrid_rank` directly, including
type-bonus and temporal decay. The retrieval module is pure Python with no
IO, so we can test it without mocking.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.domain.enums import MemoryType
from app.domain.models import MemoryNote, SearchResult
from app.domain.retrieval import (
    TYPE_BONUS,
    _temporal_decay_factor,
    hybrid_rank,
)

# Tests use zero vectors (BM25-only path), so the actual dimension is
# irrelevant to correctness. Use a small explicit constant — NOT the real
# embedding dim — to make clear this is a test fixture, not a config value.
_TEST_DIM = 16
_ZEROS = [0.0] * _TEST_DIM


def _note(
    id: str,
    content: str,
    memory_type: MemoryType = MemoryType.RECORD,
    days_ago: int = 0,
    embedding: list[float] | None = None,
) -> MemoryNote:
    created = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return MemoryNote(
        id=id,
        user_id="u",
        session_id="s",
        request_id="r",
        content=content,
        embedding=embedding or list(_ZEROS),
        memory_type=memory_type,
        type_specific={},
        created_at=created.isoformat(),
    )


def test_hybrid_rank_returns_top_n_in_score_order():
    notes = [
        _note("a", "apple banana mango"),
        _note("b", "orange kiwi papaya"),
        _note("c", "apple banana mango pear"),
    ]
    results = hybrid_rank(
        notes=notes,
        query_vec=list(_ZEROS),
        query_text="apple banana mango",
        top_n=3,
    )
    assert len(results) == 3
    # apple/banana/mango docs MUST rank ahead of orange/kiwi/papaya doc.
    assert results[2].id == "b"
    # Scores non-increasing.
    for r1, r2 in zip(results, results[1:]):
        assert r1.score >= r2.score


def test_type_bonus_promotes_matching_type():
    """With explicit type weights, matching-type memory gets bonus."""
    a = _note("a", "alpha content", memory_type=MemoryType.EVENT)
    b = _note("b", "alpha content", memory_type=MemoryType.RECORD)
    # Without weights: tied (alpha content the same).
    no_weights = hybrid_rank(
        notes=[a, b], query_vec=list(_ZEROS), query_text="alpha", top_n=2
    )
    # With type weights favoring EVENT: a should rank first.
    weighted = hybrid_rank(
        notes=[a, b],
        query_vec=list(_ZEROS),
        query_text="alpha",
        top_n=2,
        type_weights={MemoryType.EVENT: 1.0, MemoryType.RECORD: 0.0},
    )
    assert weighted[0].id == "a"
    assert weighted[0].score >= no_weights[0].score + TYPE_BONUS - 1e-9


def test_type_bonus_with_zero_weight_does_not_promote():
    a = _note("a", "alpha content", memory_type=MemoryType.EVENT)
    b = _note("b", "alpha content", memory_type=MemoryType.RECORD)
    no_weights = hybrid_rank(
        notes=[a, b], query_vec=list(_ZEROS), query_text="alpha", top_n=2
    )
    # If weights are provided but neither type is favored, no bonus applies.
    null_weights = hybrid_rank(
        notes=[a, b],
        query_vec=list(_ZEROS),
        query_text="alpha",
        top_n=2,
        type_weights={MemoryType.EVENT: 0.0, MemoryType.RECORD: 0.0},
    )
    assert null_weights[0].score == no_weights[0].score


def test_temporal_decay_factor_decays_old_memories():
    now = datetime.now(timezone.utc)
    fresh = _temporal_decay_factor(now.isoformat(), now)
    very_old = _temporal_decay_factor(
        (now - timedelta(days=365)).isoformat(), now
    )
    assert fresh > very_old
    # Floor: even very old memories get at least 0.5.
    assert very_old >= 0.5


def test_hybrid_rank_respects_top_n():
    notes = [
        _note(f"n{i}", f"unique content {i}") for i in range(10)
    ]
    results = hybrid_rank(
        notes=notes, query_vec=list(_ZEROS), query_text="any", top_n=3
    )
    assert len(results) == 3


def test_hybrid_rank_empty_input():
    results = hybrid_rank(
        notes=[], query_vec=list(_ZEROS), query_text="anything", top_n=10
    )
    assert results == []


def test_hybrid_rank_includes_memory_type_in_result():
    notes = [
        _note("x", "alpha", memory_type=MemoryType.PROFILE),
        _note("y", "alpha", memory_type=MemoryType.EVENT),
    ]
    results = hybrid_rank(
        notes=notes, query_vec=list(_ZEROS), query_text="alpha", top_n=2
    )
    assert {r.memory_type for r in results} == {MemoryType.PROFILE, MemoryType.EVENT}


def test_hybrid_rank_can_disable_temporal_decay():
    # Build a corpus large enough that BM25 has discriminative scores.
    now = datetime.now(timezone.utc)
    notes = [
        _note("query", "apple banana mango", days_ago=0),
        _note("fresh", "apple banana mango", days_ago=0),
        _note("old", "apple banana mango", days_ago=365),
        _note("noise1", "kiwi papaya dragonfruit", days_ago=0),
        _note("noise2", "orange pineapple watermelon", days_ago=0),
        _note("noise3", "grapefruit lemon lime", days_ago=0),
        _note("noise4", "strawberry blueberry raspberry", days_ago=0),
        _note("noise5", "cherry peach apricot", days_ago=0),
    ]
    with_decay = hybrid_rank(
        notes=notes, query_vec=list(_ZEROS), query_text="apple banana mango",
        top_n=3, apply_temporal_decay=True,
    )
    without_decay = hybrid_rank(
        notes=notes, query_vec=list(_ZEROS), query_text="apple banana mango",
        top_n=3, apply_temporal_decay=False,
    )
    # Without decay, fresh and old have the same score (same content).
    fresh_no = next(r for r in without_decay if r.id == "fresh")
    old_no = next(r for r in without_decay if r.id == "old")
    assert fresh_no.score == old_no.score
    # With decay, fresh > old.
    fresh = next(r for r in with_decay if r.id == "fresh")
    old = next(r for r in with_decay if r.id == "old")
    assert fresh.score > old.score


def test_temporal_decay_factor_is_lower_for_older():
    """Direct unit test of the decay function (no corpus dependency)."""
    from app.domain.retrieval import _temporal_decay_factor

    now = datetime.now(timezone.utc)
    fresh = _temporal_decay_factor(
        (now - timedelta(days=1)).isoformat(), now
    )
    old = _temporal_decay_factor(
        (now - timedelta(days=200)).isoformat(), now
    )
    assert fresh > old


def test_temporal_decay_factor_has_floor():
    """Even very old memories should not decay below 0.5 (avoids disappearing)."""
    from app.domain.retrieval import _temporal_decay_factor

    now = datetime.now(timezone.utc)
    ancient = _temporal_decay_factor(
        (now - timedelta(days=10_000)).isoformat(), now
    )
    assert ancient >= 0.5
