"""Centralized algorithm parameters for the code-memory domain layer.

This module is the **single source of truth** for every tunable constant used
by the pure domain functions (retrieval, revision, decision-utility, symbol
graph, context gate, chunker). Domain modules must NOT hard-code magic numbers;
they accept a `DomainParams` instance (defaulting to `DEFAULT_PARAMS`) so the
service layer can wire env-backed config without leaking IO into domain.

Design rules (per project convention):
  - `RANDOM_SEED` is the first symbol in the file and is used by every
    sampling / shuffle / PageRank initialization.
  - All weights/thresholds live here as named fields with docstrings.
  - Defaults are conservative, literature-grounded, and overridable.
"""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# GLOBAL RANDOM SEED — must remain 20234150 per project convention.
# Every stochastic operation (BM25 tie-break shuffle, PageRank reset vector,
# LLM batch ordering) draws from a RNG seeded with this value for reproducible
# behaviour across runs.
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 20234150


@dataclass(frozen=True)
class DomainParams:
    """All tunable algorithm parameters consumed by the domain layer.

    Frozen so it can be safely shared across calls. The service layer builds
    an instance from `app.config.Settings`; tests construct bespoke instances.
    """

    # --- Retrieval (Stage 1 recall) -----------------------------------------
    top_k: int = 100
    # Stage 1 surfaces top_k * recall_multiplier candidates before Stage 2
    # reranks down to top_k. Higher = better recall, more LLM cost in Stage 2.
    recall_multiplier: int = 5
    # Adaptive type-bonus added to BM25 for memories matching query intent
    # (LeanMem §3.4). Multiplied by the query weight in (0, 1].
    type_bonus: float = 0.3
    # Temporal decay: half-life ≈ ln(2)/decay days. 0.0077 → ~90 days.
    temporal_decay: float = 0.0077
    # Floor so very old memories still surface on strong match.
    temporal_floor: float = 0.5

    # --- Decision-utility rerank (Stage 2, CICL) ----------------------------
    # score = w1*action_relevance + w2*failure_match + w3*scope_match
    #       + w4*recency + w5*type_weight
    w_action_relevance: float = 0.35
    w_failure_match: float = 0.25
    w_scope_match: float = 0.20
    w_recency: float = 0.10
    w_type_weight: float = 0.10

    # --- Chunker (task-boundary segmentation) -------------------------------
    chunk_max_messages: int = 30
    chunk_max_words: int = 3000
    # A new TodoWrite / long idle gap / explicit task verb starts a new segment.
    chunk_boundary_on_todo: bool = True

    # --- Revision / dedup (GEM + OpenAI dreaming) --------------------------
    near_dup_threshold: float = 0.85
    dedup_neighbor_cap: int = 100
    # How many existing neighbours to consider for merge/contradict resolution.
    revision_neighbor_topk: int = 3
    # Cosine similarity above which two cards are candidates for merge/contradict.
    revision_sim_threshold: float = 0.78

    # --- Symbol graph (Aider repomap) ---------------------------------------
    pagerank_edited_boost: float = 50.0
    pagerank_damping: float = 0.85
    pagerank_iterations: int = 40
    pagerank_top_n: int = 10
    symbol_expand_enabled: bool = True

    # --- Context gate / granularity (ContextSniper + Anthropic) -------------
    default_granularity: str = "L1"  # L0 | L1 | L2
    # Max chars for L0 one-liner.
    l0_max_chars: int = 160

    # --- LLM circuit breaker (Claude Code lesson) ---------------------------
    # Consecutive LLM failures after which the pipeline switches to zero-LLM
    # fallback for the rest of the request.
    llm_failure_threshold: int = 3
    llm_batch_size: int = 5

    # --- Transcript parsing -------------------------------------------------
    # Truncate individual tool_result text blocks to this many chars before
    # feeding to the LLM extractor (keeps prompts bounded).
    transcript_tool_result_cap: int = 1200
    transcript_max_segment_chars: int = 8000


# Module-level default instance. Domain functions use this when the caller does
# not inject a bespoke `DomainParams` (e.g. unit tests of pure logic).
DEFAULT_PARAMS = DomainParams()
