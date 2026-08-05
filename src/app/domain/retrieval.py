"""Hybrid retrieval ranking.

Combines:
  1. Dense cosine similarity (text-embedding-3-small).
  2. BM25 lexical score over a code-aware index.
  3. Adaptive type-bonus (LeanMem §3.4) that boosts memories whose
     `memory_type` matches the query-plan weights.

The primary sort key is BM25 (code identifiers are lexical signals); dense
acts as a tiebreaker. The returned `score` reflects the primary sort key so
that `score` order == `id` order in the response.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np

from . import bm25 as bm25_ops
from .enums import MemoryType
from .models import MemoryNote, SearchResult
from .tokenizer import tokenize


# Type-bonus multiplier for memories of the query's preferred type.
# Multiplied by the query's weight in (0, 1]; stays within (0, 0.3].
TYPE_BONUS = 0.3

# Mild temporal decay: half-life ~ 90 days.
# Half-life = ln(2) / decay; we want half-life ≈ 90 days → decay ≈ 0.0077.
TEMPORAL_DECAY = 0.0077


def _cosine_scores(query: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Per-row cosine similarity with safe normalize (avoids /0)."""
    qn = float(np.linalg.norm(query)) + 1e-12
    norms = np.linalg.norm(mat, axis=1) + 1e-12
    return (mat @ query) / (norms * qn)


def _age_in_days(created_at: str, now: datetime) -> float:
    """Days between the memory's creation time and `now` (clamped to >= 0)."""
    try:
        ts = datetime.fromisoformat(created_at)
    except ValueError:
        return 0.0
    if ts.tzinfo is None:
        # Treat naive timestamps as UTC for stable comparison.
        ts = ts.replace(tzinfo=now.tzinfo)
    return max(0.0, (now - ts).total_seconds() / 86400.0)


def _temporal_decay_factor(created_at: str, now: datetime) -> float:
    """Exp-decay multiplier in (0, 1]; never below 0.5 within 1 year."""
    days = _age_in_days(created_at, now)
    factor = math.exp(-TEMPORAL_DECAY * days)
    # Floor at 0.5 so very old memories still surface if they match strongly.
    return max(0.5, factor)


def hybrid_rank(
    notes: list[MemoryNote],
    query_vec: list[float],
    query_text: str,
    top_n: int,
    type_weights: dict[MemoryType, float] | None = None,
    *,
    apply_temporal_decay: bool = True,
    now: datetime | None = None,
) -> list[SearchResult]:
    """Rank `notes` for the given query and return the top-N results.

    Returns a list of `SearchResult` sorted by descending relevance. The score
    field is the primary sort value (BM25 + type bonus + temporal decay).

    Args:
        notes: candidate memories (already filtered by user_id).
        query_vec: dense embedding of the enhanced query.
        query_text: enhanced query text used for BM25 tokenization.
        top_n: maximum number of results to return.
        type_weights: optional adaptive retrieval weights (LeanMem §3.4).
        apply_temporal_decay: if True, multiply final score by exp-decay.
        now: override "current time" (used by tests for determinism).
    """
    if top_n <= 0 or not notes:
        return []

    now = now or datetime.now().astimezone()
    n = len(notes)
    dim = len(query_vec)
    mat = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
    # Build dense matrix in one shot; decode embeddings from raw bytes lazily.
    dense_matrix = np.zeros((n, dim), dtype=np.float32)
    for i, note in enumerate(notes):
        dense_matrix[i] = np.asarray(note.embedding, dtype=np.float32)
    dense_scores = _cosine_scores(dense_matrix[0], dense_matrix)

    query_tokens = tokenize(query_text)
    if query_tokens:
        tokenized_docs = [tokenize(note.content) for note in notes]
        index = bm25_ops.build_index(tokenized_docs)
        if index is not None:
            bm25_scores = bm25_ops.score_query(index, query_tokens)
        else:
            bm25_scores = np.zeros(n, dtype=np.float64)
    else:
        # No lexical signal: fall back to dense primary.
        bm25_scores = np.zeros(n, dtype=np.float64)

    # Primary sort value: BM25 (with optional type-bonus + temporal decay).
    primary = bm25_scores.astype(np.float64).copy()
    if type_weights:
        for i, note in enumerate(notes):
            w = type_weights.get(note.memory_type, 0.0)
            if w > 0:
                primary[i] += TYPE_BONUS * w
    # Shift to non-negative BEFORE applying temporal decay; otherwise multiplying
    # negative scores by a 0-1 factor would make them appear *better*.
    if primary.size:
        primary = primary - primary.min()
    if apply_temporal_decay:
        for i, note in enumerate(notes):
            primary[i] *= _temporal_decay_factor(note.created_at, now)

    # Order: primary descending; dense score as tiebreaker (stable).
    order = np.lexsort((-dense_scores, -primary))

    top_n = min(top_n, n)
    return [
        SearchResult(
            id=notes[i].id,
            content=notes[i].content,
            score=float(primary[i]),
            created_at=notes[i].created_at,
            memory_type=notes[i].memory_type,
        )
        for i in order[:top_n]
    ]
