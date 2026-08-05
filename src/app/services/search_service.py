"""Search pipeline: two-stage retrieval → decision-utility rerank → gate.

Stage 1 (Recall): hybrid BM25 + dense + type-bonus + temporal decay surfaces
`top_k * recall_multiplier` candidates. When the symbol graph is populated,
retrieved RECORD memories seed a personalized PageRank that expands the
candidate set with structurally related files/symbols (Aider repomap) — the
cross-file localization signal that pure vector retrieval misses.

Stage 2 (Decision-utility rerank, CICL): candidates are re-sorted by how much
they would CHANGE the agent's next action (action_relevance, failure_match,
scope_match, recency, type_weight) rather than by raw semantic similarity.
An LLM scorer is used when available; otherwise a deterministic lexical
heuristic. The top `top_k` survive.

Context gate (ContextSniper + Anthropic): intent-aware filtering drops
off-target types, then each survivor is packaged at the configured granularity
(L0 one-liner / L1 structured 5 fields / L2 full content) for just-in-time
retrieval.

Per-request LLM circuit breaker is reset at entry.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Sequence

from ..config import get_domain_params, get_settings
from ..domain import prompts
from ..domain.context_gate import GatedResult, gate
from ..domain.decision_utility import UtilityScores, heuristic_utility, rerank_by_utility
from ..domain.enums import MemoryType
from ..domain.experience_card import ExperienceCard
from ..domain.models import MemoryNote, SearchResult
from ..domain.retrieval import hybrid_rank
from ..domain.symbol_graph import build_graph, expand_files
from ..infrastructure import embed as embed_client
from ..infrastructure import llm as llm_client
from ..infrastructure import sqlite as db


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query understanding (LLM + heuristic fallback)
# ---------------------------------------------------------------------------

_DEFAULT_TYPE_WEIGHTS: dict[MemoryType, float] = {}


def _enhance_query(query: str, options: Sequence[str] | None) -> str:
    """Reduce the query to a retrieval-friendly text. Falls back to raw query."""
    opts_block = ""
    if options:
        opts_block = "\n\nOptions:\n" + "\n".join(options)
    result = llm_client.chat_text(
        system=prompts.ENHANCE_QUERY_PROMPT,
        user=query + opts_block,
        temperature=0.0,
    )
    return result.strip() if result.strip() else query


def _classify_query_intent(query: str) -> dict[MemoryType, float]:
    """Return a mapping memory_type → weight in [0, 1]. Empty dict = no bias."""
    result = llm_client.chat_json(
        system=prompts.CLASSIFY_QUERY_INTENT_PROMPT,
        user=query,
        temperature=0.0,
    )
    if not result or not isinstance(result, dict):
        return dict(_DEFAULT_TYPE_WEIGHTS)
    types = result.get("types") or []
    weights = result.get("weights") or []
    out: dict[MemoryType, float] = {}
    for t, w in zip(types, weights):
        if not isinstance(t, str) or not isinstance(w, (int, float)):
            continue
        try:
            mt = MemoryType(t.strip().lower())
        except ValueError:
            continue
        out[mt] = max(0.0, min(1.0, float(w)))
    return out


# ---------------------------------------------------------------------------
# LLM decision-utility scorer (injected into the pure rerank module)
# ---------------------------------------------------------------------------

def _make_utility_scorer():
    """Build an LLM-backed utility scorer; None → heuristic is used."""
    if not llm_client.client_available():
        return None

    def scorer(query: str, note: MemoryNote, card: ExperienceCard) -> tuple[float, float, float, float]:
        if llm_client.breaker_tripped():
            u = heuristic_utility(query, note, card, None, datetime.now().astimezone())
            return u.action_relevance, u.failure_match, u.scope_match, u.recency
        candidate_text = (
            f"trigger: {card.trigger}\nroot_cause: {card.root_cause}\n"
            f"action_hint: {card.action_hint}\nfailure_if_ignored: {card.failure_if_ignored}\n"
            f"scope: {card.scope}\ntemporal_anchor: {card.temporal_anchor}"
        )
        result = llm_client.chat_json(
            system=prompts.DECISION_UTILITY_PROMPT,
            user=f"Task query:\n{query[:2000]}\n\nCandidate memory:\n{candidate_text[:2000]}",
            temperature=0.0,
        )
        if not result or not isinstance(result, dict):
            u = heuristic_utility(query, note, card, None, datetime.now().astimezone())
            return u.action_relevance, u.failure_match, u.scope_match, u.recency
        return (
            float(result.get("action_relevance", 0.3) or 0.3),
            float(result.get("failure_match", 0.3) or 0.3),
            float(result.get("scope_match", 0.3) or 0.3),
            float(result.get("recency", 0.3) or 0.3),
        )
    return scorer


# ---------------------------------------------------------------------------
# Symbol-graph expansion (Aider repomap)
# ---------------------------------------------------------------------------

# File-path-like tokens in the query (e.g. "auth.py", "django/contrib/admin").
_FILE_TOKEN_RE = re.compile(r"[\w./\-]+\.[A-Za-z]{1,4}\b|[A-Za-z_][A-Za-z0-9_./\-]{2,}")


def _query_seed_files(query: str) -> list[str]:
    """Pull file-path-like tokens out of the query to seed PageRank."""
    return [m.group(0) for m in _FILE_TOKEN_RE.finditer(query)][:20]


def _symbol_expand(
    user_id: str,
    query: str,
    recalled: list[SearchResult],
    notes_by_id: dict[str, MemoryNote],
    params,
) -> list[SearchResult]:
    """Expand the candidate set with structurally related memories.

    Builds the user's symbol graph from stored `file_symbols`, seeds PageRank
    with the query's file tokens + the recalled memories' files_symbols, then
    boosts memories that reference the PageRank-related files. No-op (returns
    recalled unchanged) when the graph is empty or the feature is disabled.
    """
    if not params.symbol_expand_enabled or not recalled:
        return recalled

    refs = db.fetch_file_symbols(user_id)
    if not refs:
        return recalled
    graph = build_graph(refs)
    if not graph.files:
        return recalled

    # Seed: query file tokens + every recalled memory's files_symbols.
    seed: set[str] = set(_query_seed_files(query))
    for r in recalled[: params.recall_multiplier * 10]:
        note = notes_by_id.get(r.id)
        if note is None:
            continue
        card = ExperienceCard.from_type_specific(note.memory_type, note.type_specific)
        seed.update(card.files_symbols)
    if not seed:
        return recalled

    ranked_files = expand_files(graph, seed, params)
    if not ranked_files:
        return recalled
    related_files = {rf.file for rf in ranked_files}
    related_symbols = {s for rf in ranked_files for s in rf.symbols}

    # Boost memories whose files_symbols intersect the related set. The boost
    # is a fixed additive nudge on the Stage-1 score (kept small so it acts as
    # a tiebreaker rather than overwhelming strong lexical matches).
    SYMBOL_BOOST = 0.15
    boosted: list[SearchResult] = []
    for r in recalled:
        note = notes_by_id.get(r.id)
        if note is not None:
            card = ExperienceCard.from_type_specific(note.memory_type, note.type_specific)
            hit = bool(set(card.files_symbols) & related_files) or bool(
                set(card.files_symbols) & related_symbols
            )
            if hit:
                boosted.append(SearchResult(
                    id=r.id, content=r.content, memory_type=r.memory_type,
                    score=r.score + SYMBOL_BOOST, created_at=r.created_at,
                ))
                continue
        boosted.append(r)
    return boosted


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def search(
    user_id: str,
    query: str,
    top_k: int,
    options: Sequence[str] | None = None,
    *,
    use_llm: bool = True,
) -> list[GatedResult]:
    """Run the Search pipeline. Returns the top-K gated results."""
    settings = get_settings()
    params = get_domain_params()
    llm_client.reset_breaker()

    # 1. Candidate memories for this user (excludes superseded old versions).
    notes = db.fetch_memories_by_user(user_id)
    if not notes:
        return []
    notes_by_id: dict[str, MemoryNote] = {n.id: n for n in notes}

    # 2. Query understanding (LLM or fallback to raw query / no type bias).
    use_llm = use_llm and settings.use_llm_on_search
    if use_llm and llm_client.client_available():
        enhanced = _enhance_query(query, options)
        type_weights = _classify_query_intent(query)
    else:
        enhanced = query
        type_weights = dict(_DEFAULT_TYPE_WEIGHTS)

    # 3. Embed the (enhanced) query.
    query_vec = embed_client.embed_texts([enhanced])[0] if enhanced else []

    # 4. Stage 1 recall: over-fetch by recall_multiplier, then symbol-expand.
    recall_k = max(top_k, min(top_k * params.recall_multiplier, len(notes)))
    candidates = hybrid_rank(
        notes=notes,
        query_vec=query_vec,
        query_text=enhanced,
        top_n=recall_k,
        type_weights=type_weights or None,
    )
    candidates = _symbol_expand(user_id, enhanced, candidates, notes_by_id, params)

    # 5. Stage 2 decision-utility rerank (CICL).
    scorer = _make_utility_scorer() if use_llm else None
    reranked = rerank_by_utility(
        query=enhanced,
        candidates=candidates,
        notes_by_id=notes_by_id,
        type_weights=type_weights or None,
        params=params,
        utility_scorer=scorer,
    )[:top_k]

    # 6. Context gate: intent filter + granularity packaging (configured internally).
    return gate(
        query_intent_weights=type_weights or None,
        results=reranked,
        notes_by_id=notes_by_id,
        granularity=params.default_granularity,
        params=params,
        keep_min=min(top_k, 1),
    )
