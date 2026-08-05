"""BM25 scoring wrapped in a small functional API.

The `rank_bm25` library is the de-facto standard scorer for BM25 in Python;
we wrap it domain-side so the rest of the system never imports `rank_bm25`
directly. This keeps the dependency at the seam.
"""
from __future__ import annotations

import numpy as np

from rank_bm25 import BM25Okapi

from .tokenizer import tokenize


def build_index(tokenized_docs: list[list[str]]) -> BM25Okapi | None:
    """Build a BM25Okapi index from already-tokenized documents.

    Returns None for empty input so callers can short-circuit cleanly.
    """
    if not tokenized_docs:
        return None
    return BM25Okapi(tokenized_docs)


def score_query(index: BM25Okapi | None, query_tokens: list[str]) -> np.ndarray:
    """Return BM25 scores as a float64 numpy array aligned with `index` docs.

    When `index` is None (empty corpus), returns an empty array.
    """
    if index is None:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(index.get_scores(query_tokens), dtype=np.float64)


def rank_documents(index: BM25Okapi | None, query_tokens: list[str], top_n: int) -> list[int]:
    """Return indices of the top-N documents sorted by descending BM25 score.

    Uses `argpartition` for O(N) instead of full O(N log N) sort when N > top_n.

    Note: standard BM25 IDF can become negative when a term appears in more
    than half the corpus (i.e., df > N/2). We shift scores so that the
    minimum score is 0; this preserves relative ordering while guaranteeing
    non-negative values for downstream comparisons.
    """
    scores = score_query(index, query_tokens)
    n = len(scores)
    if n == 0:
        return []
    # Shift to non-negative while preserving ordering.
    scores = scores - scores.min()
    top_n = max(1, min(top_n, n))
    if top_n >= n:
        return list(np.argsort(-scores))
    partition = np.argpartition(-scores, top_n - 1)[:top_n]
    return partition[np.argsort(-scores[partition])].tolist()
