"""Stage-2 decision-utility rerank (CICL).

Stage 1 (retrieval) surfaces candidates by *semantic* relevance. Stage 2
reranks by *decision utility*: how much would this memory CHANGE the agent's
next action? A memory that merely mentions the same keyword but offers no
actionable hint is demoted; one whose action_hint maps to the query's
suspected root cause is promoted.

Score = w1*action_relevance + w2*failure_match + w3*scope_match
      + w4*recency + w5*type_weight

The four utility axes are produced either by an LLM (DECISION_UTILITY_PROMPT,
injected as `utility_scorer`) or by a zero-LLM heuristic based on lexical
overlap + scope/repo match + temporal recency. The heuristic keeps the
pipeline functional without an API key and is fully deterministic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Sequence

from .enums import MemoryType
from .experience_card import ExperienceCard
from .models import MemoryNote, SearchResult
from .params import DomainParams, DEFAULT_PARAMS


@dataclass
class UtilityScores:
    action_relevance: float = 0.0
    failure_match: float = 0.0
    scope_match: float = 0.0
    recency: float = 0.0
    type_weight: float = 0.0

    def weighted(self, p: DomainParams) -> float:
        return (
            p.w_action_relevance * self.action_relevance
            + p.w_failure_match * self.failure_match
            + p.w_scope_match * self.scope_match
            + p.w_recency * self.recency
            + p.w_type_weight * self.type_weight
        )


# A utility scorer takes (query, candidate_note, candidate_card) and returns
# the four LLM-judged axes (action_relevance, failure_match, scope_match,
# recency). The service layer wires this to the LLM; tests inject a stub.
UtilityScorer = Callable[[str, MemoryNote, ExperienceCard], tuple[float, float, float, float]]


# ---------------------------------------------------------------------------
# Zero-LLM heuristic scorer
# ---------------------------------------------------------------------------

def _token_set(text: str) -> set[str]:
    return {t.lower() for t in text.split() if len(t) > 1} if text else set()


def _scope_tokens(scope: str) -> set[str]:
    """Split a scope string 'repo/lang#scenario' into individual tokens."""
    if not scope:
        return set()
    import re as _re
    return {t.lower() for t in _re.split(r"[/#\-_.]+", scope) if len(t) > 1}


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _recency_from_created(created_at: str, now: datetime) -> float:
    """Exp-decay recency in [0, 1]; 1.0 today, ~0.5 at 90 days, floor 0.1."""
    try:
        ts = datetime.fromisoformat(created_at)
    except (ValueError, TypeError):
        return 0.3
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=now.tzinfo)
    days = max(0.0, (now - ts).total_seconds() / 86400.0)
    return max(0.1, math.exp(-0.0077 * days))


def heuristic_utility(
    query: str,
    note: MemoryNote,
    card: ExperienceCard,
    type_weights: dict[MemoryType, float] | None,
    now: datetime,
) -> UtilityScores:
    """Deterministic utility estimate without an LLM."""
    q_tokens = _token_set(query)
    # action_relevance: overlap of query with action_hint + gist (the actionable text).
    actionable = " ".join([card.action_hint, card.gist, note.content[:400]])
    action_rel = _overlap(q_tokens, _token_set(actionable))
    # failure_match: overlap of query with failure_if_ignored + trigger.
    failure_text = " ".join([card.failure_if_ignored, card.trigger])
    failure = _overlap(q_tokens, _token_set(failure_text))
    # scope_match: does the query mention the repo/language/scenario in scope?
    scope = 0.0
    if card.scope:
        scope_tokens = _scope_tokens(card.scope)
        if scope_tokens & q_tokens:
            scope = 1.0
        else:
            # Partial: any scope token appears anywhere in the query.
            ql = query.lower()
            scope = 0.3 if any(t in ql for t in scope_tokens) else 0.0
    recency = _recency_from_created(note.created_at, now)
    tw = (type_weights or {}).get(note.memory_type, 0.0)
    return UtilityScores(
        action_relevance=action_rel,
        failure_match=failure,
        scope_match=scope,
        recency=recency,
        type_weight=tw,
    )


# ---------------------------------------------------------------------------
# Rerank
# ---------------------------------------------------------------------------

def rerank_by_utility(
    query: str,
    candidates: Sequence[SearchResult],
    notes_by_id: dict[str, MemoryNote],
    type_weights: dict[MemoryType, float] | None,
    params: DomainParams = DEFAULT_PARAMS,
    *,
    utility_scorer: UtilityScorer | None = None,
    now: datetime | None = None,
) -> list[SearchResult]:
    """Rerank Stage-1 candidates by decision utility.

    Returns a new list sorted by descending utility score. The `score` field of
    each result is replaced with the utility score (Stage-2's view). Candidates
    lacking a card fall back to a neutral score so they are not unfairly buried.
    """
    now = now or datetime.now().astimezone()
    scored: list[tuple[float, float, SearchResult]] = []

    for cand in candidates:
        note = notes_by_id.get(cand.id)
        if note is None:
            # Without the underlying note we cannot score utility; keep as-is
            # with a neutral utility so Stage-1 score orders these.
            scored.append((0.0, cand.score, cand))
            continue
        card = ExperienceCard.from_type_specific(note.memory_type, note.type_specific)
        if utility_scorer is not None:
            try:
                ar, fm, sm, rc = utility_scorer(query, note, card)
            except Exception:  # noqa: BLE001
                u = heuristic_utility(query, note, card, type_weights, now)
                ar, fm, sm, rc = u.action_relevance, u.failure_match, u.scope_match, u.recency
            tw = (type_weights or {}).get(note.memory_type, 0.0)
            u = UtilityScores(ar, fm, sm, rc, tw)
        else:
            u = heuristic_utility(query, note, card, type_weights, now)
        utility = u.weighted(params)
        # Stage-2 utility is the PRIMARY sort key; Stage-1 score is the
        # tiebreaker (lexsort: primary desc, stage1 desc). We do NOT blend,
        # because the whole point of Stage-2 is to override Stage-1 misses.
        scored.append((utility, cand.score, cand))

    # Sort by utility desc, then stage1 score desc.
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return [r for _, _, r in scored]
