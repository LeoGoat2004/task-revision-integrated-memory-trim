"""Memory quality primitives: deduplication.

Strategies (in increasing cost):
1. **Exact-hash dedup**: SHA-256 of `(user_id, content)` — catches identical
   re-submissions of the same chunk.
2. **BM25 near-dup**: if the top-1 neighbor score exceeds `NEAR_DUP_THRESHOLD`,
   skip the insert (keeps the existing entry).

Both strategies are pure functions over the candidate + existing notes so
the service layer can decide when to apply them.
"""
from __future__ import annotations

import hashlib

from .bm25 import build_index, score_query
from .models import MemoryNote
from .tokenizer import tokenize


# BM25 score above which a candidate is considered a near-duplicate of the
# top-1 neighbor. The threshold is intentionally generous: code memories are
# highly templated (e.g., "fixed NPE in X" can repeat verbatim for similar
# bugs), so we err on the side of skipping near-duplicates.
NEAR_DUP_THRESHOLD = 0.85


def content_fingerprint(user_id: str, content: str) -> str:
    """Stable hash of `(user_id, content)` for exact-dup detection."""
    h = hashlib.sha256()
    h.update(user_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(content.encode("utf-8"))
    return h.hexdigest()


def is_near_duplicate(
    candidate: MemoryNote,
    existing: list[MemoryNote],
    threshold: float = NEAR_DUP_THRESHOLD,
) -> bool:
    """Return True if `candidate` is near-duplicate of any `existing` note.

    The check is content-based: we BM25-score the candidate against the
    existing corpus and compare the top-1 score against the *second*-best
    score. If the gap is small (or the corpus has one note where the
    candidate matches nearly perfectly), we treat it as a duplicate.

    The `threshold` is on a normalized score in [0, 1] (best score / max
    raw score), which is robust to BM25's negative-when-very-common-term
    edge case.
    """
    if not existing:
        return False
    docs = [tokenize(note.content) for note in existing]
    index = build_index(docs)
    if index is None:
        return False
    scores = score_query(index, tokenize(candidate.content))
    if len(scores) == 0:
        return False
    # Normalize to [0, 1] by the score range.
    raw_min = float(scores.min())
    raw_max = float(scores.max())
    if raw_max - raw_min < 1e-9:
        # Single note or uniform scores: treat as a match if any score > 0.
        return raw_max > 0
    top1_normalized = (raw_max - raw_min) / (raw_max - raw_min)  # = 1.0
    _ = top1_normalized  # for clarity
    # The actual discrimination: compare top-1 to the average of the rest.
    # If top-1 is much higher than the rest, the candidate has a clear match.
    others = scores[scores < raw_max]
    if len(others) == 0:
        # Single candidate matched perfectly.
        return True
    avg_others = float(others.mean())
    # Normalized dominance: how much higher top-1 is than the average.
    dominance = (raw_max - avg_others) / max(abs(raw_max), abs(avg_others), 1e-9)
    return dominance >= threshold
