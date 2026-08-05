"""Revision / ingestion logic: dedup, merge, contradiction resolution.

Implements the GEM/MemState "ingestion" operator and OpenAI Dreaming V3's
supports/refines/contradicts relation model. The Add pipeline is no longer a
blind append: every candidate card is reconciled against existing neighbours
so the memory base stays non-redundant and self-consistent.

Flow (driven by the service layer):
  1. `exact_duplicate` — SHA-256 of (user_id, content). Hit → SKIP.
  2. `find_nearest` — BM25 + dense cosine over existing notes. If the best
     neighbour's combined similarity ≥ `revision_sim_threshold` → candidate.
  3. `judge_relation` — LLM (RELATION_JUDGE_PROMPT) or zero-LLM heuristic:
       duplicate   → SKIP
       supports    → MERGE (corroborating detail)
       refines     → MERGE (new fields override existing)
       contradicts → SUPERSEDE (new version supersedes old; keep version chain)
       unrelated   → INSERT
  4. `merge_cards` — apply merged_fields onto the existing card.

Pure domain: the LLM call is injected as a `relation_judge` callback so this
module stays IO-free and unit-testable. The zero-LLM fallback merges on high
similarity and never supersedes (conservative — avoids data loss without an
LLM to confirm a genuine contradiction).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

import numpy as np

from .experience_card import ExperienceCard
from .models import MemoryNote
from .params import DomainParams, DEFAULT_PARAMS
from .quality import content_fingerprint


class RevisionAction(str, Enum):
    SKIP = "skip"           # exact / judged duplicate
    INSERT = "insert"       # no near-neighbour; new entry
    MERGE = "merge"         # supports/refines → fold into existing
    SUPERSEDE = "supersede"  # contradicts → new version replaces old


@dataclass
class RevisionDecision:
    action: RevisionAction
    target_id: str = ""           # existing note id for MERGE / SUPERSEDE
    merged_card: ExperienceCard | None = None  # merged card for MERGE
    reason: str = ""

    @property
    def persists(self) -> bool:
        """Whether the candidate results in a write (insert/merge/supersede)."""
        return self.action != RevisionAction.SKIP


# A relation judge takes (candidate_text, existing_text) and returns one of
# the relation strings plus an optional merged-fields dict. The service layer
# implements this with an LLM call; tests inject a stub.
RelationJudge = Callable[[str, str], tuple[str, dict]]


# ---------------------------------------------------------------------------
# Step 1: exact duplicate
# ---------------------------------------------------------------------------

def exact_duplicate(
    user_id: str, content: str, existing: Sequence[MemoryNote]
) -> bool:
    """True if (user_id, content) SHA-256 matches an existing note."""
    fp = content_fingerprint(user_id, content)
    return any(content_fingerprint(user_id, n.content) == fp for n in existing)


# ---------------------------------------------------------------------------
# Step 2: nearest neighbour (BM25 + dense cosine)
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(av)) + 1e-12
    nb = float(np.linalg.norm(bv)) + 1e-12
    return float(np.dot(av, bv) / (na * nb))


def find_nearest(
    candidate_content: str,
    candidate_embedding: list[float],
    existing: Sequence[MemoryNote],
    params: DomainParams = DEFAULT_PARAMS,
) -> MemoryNote | None:
    """Return the single most-similar existing note, or None if none clear the bar.

    Combined similarity = 0.5 * cosine + 0.5 * bm25_norm. We keep this cheap
    (no index build) because `existing` is already capped to
    `dedup_neighbor_cap` by the service layer.
    """
    if not existing:
        return None

    # BM25 over the candidate vs each existing content.
    from .tokenizer import tokenize
    from . import bm25 as bm25_ops

    cand_tokens = tokenize(candidate_content)
    docs = [tokenize(n.content) for n in existing]
    index = bm25_ops.build_index(docs)
    if index is not None:
        raw = bm25_ops.score_query(index, cand_tokens)
    else:
        raw = np.zeros(len(existing), dtype=np.float64)
    # BM25 IDF can go negative for small corpora (df > N/2), and a single-doc
    # corpus collapses all scores to one value. Shift to non-negative, then
    # normalize by the max; when even the shift is degenerate (all-equal
    # scores, e.g. a one-note memory base), fall back to a binary "shares a
    # non-trivial token with the candidate" signal so a matching neighbour is
    # still surfaced instead of being buried as "no near neighbour".
    if raw.size:
        shifted = raw - raw.min()
        mx = float(shifted.max())
        if mx > 1e-9:
            bm25_norm = shifted / mx
        else:
            cand_set = set(cand_tokens)
            bm25_norm = np.array(
                [1.0 if (cand_set & set(d)) else 0.0 for d in docs],
                dtype=np.float64,
            )
    else:
        bm25_norm = np.zeros(0, dtype=np.float64)

    best: MemoryNote | None = None
    best_score = -1.0
    for i, note in enumerate(existing):
        cos = _cosine(candidate_embedding, note.embedding)
        bm25_n = float(bm25_norm[i]) if i < len(bm25_norm) else 0.0
        # Near-neighbour semantics: a candidate qualifies if EITHER signal is
        # strong (semantic similarity OR lexical overlap). Using `max` means a
        # high cosine alone (same experience, different wording) or a high BM25
        # alone (same identifiers) is enough — averaging would miss the case
        # where embeddings match but token overlap is low.
        # If embeddings are zero (no API key), max falls back to BM25-only.
        combined = max(cos, bm25_n)
        if combined > best_score:
            best_score = combined
            best = note

    if best is None:
        return None
    return best if best_score >= params.revision_sim_threshold else None


# ---------------------------------------------------------------------------
# Step 3: relation judgement + decision
# ---------------------------------------------------------------------------

def _heuristic_relation(candidate_text: str, existing_text: str) -> tuple[str, dict]:
    """Zero-LLM relation fallback. Conservative: merge on overlap, never supersede."""
    if not candidate_text or not existing_text:
        return "unrelated", {}
    # Token overlap (Jaccard) as a cheap proxy.
    a = set(candidate_text.lower().split())
    b = set(existing_text.lower().split())
    if not a or not b:
        return "unrelated", {}
    overlap = len(a & b) / len(a | b)
    if overlap >= 0.85:
        return "duplicate", {}
    if overlap >= 0.4:
        return "supports", {}
    return "unrelated", {}


def decide_revision(
    candidate_content: str,
    candidate_embedding: list[float],
    candidate_card: ExperienceCard,
    existing: Sequence[MemoryNote],
    params: DomainParams = DEFAULT_PARAMS,
    relation_judge: RelationJudge | None = None,
    user_id: str = "",
) -> RevisionDecision:
    """Decide how to ingest a candidate card against existing notes.

    Args:
        user_id: scope for the exact-hash dedup check (matches the candidate's
            future user_id). Defaults to "" which matches the legacy behaviour.
        relation_judge: optional LLM-backed judge. If None, the zero-LLM
            heuristic is used (never supersedes — safe default without an LLM).
    """
    # 1. Exact dup → skip.
    if exact_duplicate(user_id, candidate_content, existing):
        return RevisionDecision(action=RevisionAction.SKIP, reason="exact hash dup")

    # 2. Nearest neighbour.
    nearest = find_nearest(candidate_content, candidate_embedding, existing, params)
    if nearest is None:
        return RevisionDecision(action=RevisionAction.INSERT, reason="no near neighbour")

    # 3. Judge relation.
    judge = relation_judge or _heuristic_relation
    try:
        relation, merged_fields = judge(candidate_content, nearest.content)
    except Exception:  # noqa: BLE001 — never let a judge error abort ingestion
        relation, merged_fields = "unrelated", {}

    if relation == "duplicate":
        return RevisionDecision(action=RevisionAction.SKIP, target_id=nearest.id, reason="judged duplicate")
    if relation in ("supports", "refines"):
        existing_card = ExperienceCard.from_type_specific(nearest.memory_type, nearest.type_specific)
        merged = merge_cards(existing_card, candidate_card, merged_fields, override=(relation == "refines"))
        return RevisionDecision(
            action=RevisionAction.MERGE, target_id=nearest.id, merged_card=merged, reason=relation,
        )
    if relation == "contradicts":
        # Only an LLM judge can declare a contradiction (heuristic never does).
        return RevisionDecision(
            action=RevisionAction.SUPERSEDE, target_id=nearest.id, reason="contradicts",
        )
    return RevisionDecision(action=RevisionAction.INSERT, reason="unrelated despite similarity")


# ---------------------------------------------------------------------------
# Step 4: merge
# ---------------------------------------------------------------------------

# Fields that, when non-empty in the new card or merged_fields, override the
# existing card under a "refines" relation.
_MERGEABLE_FIELDS = (
    "trigger", "root_cause", "action_hint", "failure_if_ignored", "scope",
)


def merge_cards(
    existing: ExperienceCard,
    new: ExperienceCard,
    merged_fields: dict | None,
    *,
    override: bool,
) -> ExperienceCard:
    """Merge `new` into `existing`. `override=True` (refines) lets new non-empty
    fields replace existing; `override=False` (supports) only fills empties.

    `merged_fields` (from the LLM judge) takes precedence over both when present.
    """
    mf = merged_fields or {}
    merged = ExperienceCard(
        memory_type=existing.memory_type,
        trigger=existing.trigger,
        root_cause=existing.root_cause,
        action_hint=existing.action_hint,
        files_symbols=list(existing.files_symbols),
        failure_if_ignored=existing.failure_if_ignored,
        scope=existing.scope,
        temporal_anchor=existing.temporal_anchor,
        gist=existing.gist,
        entities=list(existing.entities),
        keywords=list(existing.keywords),
        pairs=list(existing.pairs),
    )

    # Apply new card fields.
    for f in _MERGEABLE_FIELDS:
        new_val = getattr(new, f, "")
        if not new_val:
            continue
        cur_val = getattr(merged, f, "")
        if override or not cur_val:
            setattr(merged, f, new_val)
    # Lists: union (deduped).
    for f in ("files_symbols", "entities", "keywords", "pairs"):
        cur = list(getattr(merged, f))
        seen = set()
        uniq = []
        for x in cur + list(getattr(new, f)):
            key = repr(x)
            if key not in seen:
                seen.add(key)
                uniq.append(x)
        setattr(merged, f, uniq)
    if new.gist and (override or not merged.gist):
        merged.gist = new.gist
    if new.temporal_anchor and (override or not merged.temporal_anchor):
        merged.temporal_anchor = new.temporal_anchor

    # Apply LLM-provided merged_fields (highest precedence).
    for f in _MERGEABLE_FIELDS:
        if f in mf and mf[f]:
            setattr(merged, f, str(mf[f]))
    return merged
