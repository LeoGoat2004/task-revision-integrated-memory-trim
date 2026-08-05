"""Tests for revision, decision_utility, context_gate."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.domain.decision_utility import (
    heuristic_utility,
    rerank_by_utility,
)
from app.domain.context_gate import gate, package
from app.domain.enums import MemoryType
from app.domain.experience_card import ExperienceCard
from app.domain.models import MemoryNote, SearchResult
from app.domain.params import DomainParams
from app.domain.revision import (
    RevisionAction,
    decide_revision,
    merge_cards,
)


def _note(
    nid: str, content: str, card: ExperienceCard | None = None,
    mtype: MemoryType = MemoryType.RECORD, created_days_ago: float = 0.0,
    embedding: list[float] | None = None,
) -> MemoryNote:
    ts = (datetime.now(timezone.utc) - timedelta(days=created_days_ago)).isoformat()
    return MemoryNote(
        id=nid, user_id="u", session_id="s", request_id="r",
        content=content, embedding=embedding or [],
        memory_type=mtype,
        type_specific=card.to_type_specific() if card else {},
        created_at=ts,
    )


# ---------------------------------------------------------------------------
# revision
# ---------------------------------------------------------------------------

def test_revision_insert_when_no_existing():
    card = ExperienceCard(memory_type=MemoryType.RECORD, action_hint="add null check")
    dec = decide_revision("fix NPE in auth.py", [], card, [])
    assert dec.action == RevisionAction.INSERT


def test_revision_skip_exact_duplicate():
    card = ExperienceCard(memory_type=MemoryType.RECORD)
    existing = [_note("e1", "fix NPE in auth.py")]
    dec = decide_revision(
        "fix NPE in auth.py", [], card, existing=existing, user_id="u",
    )
    assert dec.action == RevisionAction.SKIP


def test_revision_merge_on_supports_with_llm_judge():
    card = ExperienceCard(
        memory_type=MemoryType.EVENT, action_hint="add null check",
        files_symbols=["auth.py"],
    )
    existing_card = ExperienceCard(
        memory_type=MemoryType.EVENT, trigger="NPE on login",
        action_hint="guard token", files_symbols=["auth.py"],
    )
    # Provide similar embeddings so find_nearest clears the similarity bar.
    emb = [1.0, 0.0, 0.0]
    existing = [_note("e1", "guard token in auth.py", existing_card, embedding=emb)]
    # Stub judge: returns supports with merged action_hint.
    def judge(cand, exist):
        return "supports", {"action_hint": "add null check"}
    dec = decide_revision(
        "add null check for token", emb, card, existing,
        relation_judge=judge,
    )
    assert dec.action == RevisionAction.MERGE
    assert dec.target_id == "e1"
    assert dec.merged_card is not None
    # Merged keeps existing trigger + new action_hint.
    assert dec.merged_card.trigger == "NPE on login"
    assert dec.merged_card.action_hint == "add null check"
    assert "auth.py" in dec.merged_card.files_symbols


def test_revision_supersede_on_contradicts():
    card = ExperienceCard(memory_type=MemoryType.EVENT, root_cause="actually a race condition")
    existing_card = ExperienceCard(memory_type=MemoryType.EVENT, root_cause="null pointer")
    emb = [0.9, 0.1, 0.0]
    existing = [_note("e1", "null pointer cause", existing_card, embedding=emb)]
    def judge(cand, exist):
        return "contradicts", {}
    dec = decide_revision("race condition fix", emb, card, existing, relation_judge=judge)
    assert dec.action == RevisionAction.SUPERSEDE
    assert dec.target_id == "e1"


def test_merge_cards_refine_overrides_supports_fills():
    existing = ExperienceCard(memory_type=MemoryType.EVENT, trigger="t1", action_hint="a1")
    new = ExperienceCard(memory_type=MemoryType.EVENT, action_hint="a2-better", root_cause="rc2")
    merged = merge_cards(existing, new, {}, override=True)
    assert merged.action_hint == "a2-better"
    assert merged.root_cause == "rc2"
    assert merged.trigger == "t1"  # preserved
    # supports (override=False) only fills empties
    merged_s = merge_cards(existing, new, {}, override=False)
    assert merged_s.action_hint == "a1"  # not overridden
    assert merged_s.root_cause == "rc2"  # was empty → filled


# ---------------------------------------------------------------------------
# decision_utility
# ---------------------------------------------------------------------------

def test_heuristic_utility_favors_actionable_match():
    card_match = ExperienceCard(
        memory_type=MemoryType.EVENT, action_hint="add null check for token in auth.py",
        trigger="NPE on None token", scope="myapp/python#auth",
    )
    card_off = ExperienceCard(
        memory_type=MemoryType.EVENT, action_hint="configure CORS origins",
        scope="webapp/python#cors",
    )
    now = datetime.now(timezone.utc)
    n_match = _note("m", "auth.py NPE", card_match, created_days_ago=1)
    n_off = _note("o", "cors config", card_off, created_days_ago=1)
    u_match = heuristic_utility("fix NPE token auth.py", n_match, card_match, None, now)
    u_off = heuristic_utility("fix NPE token auth.py", n_off, card_off, None, now)
    assert u_match.action_relevance > u_off.action_relevance
    assert u_match.scope_match > u_off.scope_match


def test_rerank_by_utility_reorders():
    card_match = ExperienceCard(
        memory_type=MemoryType.EVENT, action_hint="add null check for token in auth.py",
        trigger="NPE on None token", scope="myapp/python#auth",
    )
    card_off = ExperienceCard(
        memory_type=MemoryType.EVENT, action_hint="configure CORS origins",
        scope="webapp/python#cors",
    )
    now = datetime.now(timezone.utc)
    n_match = _note("m", "auth.py NPE", card_match, created_days_ago=1)
    n_off = _note("o", "cors config", card_off, created_days_ago=1)
    notes_by_id = {n_match.id: n_match, n_off.id: n_off}
    # Stage-1 returned the off-topic one first (higher lexical score by luck).
    cands = [
        SearchResult(id="o", content="cors config", score=10.0, created_at=n_off.created_at, memory_type=MemoryType.EVENT),
        SearchResult(id="m", content="auth.py NPE", score=5.0, created_at=n_match.created_at, memory_type=MemoryType.EVENT),
    ]
    reranked = rerank_by_utility(
        "fix NPE token auth.py", cands, notes_by_id, None, now=now,
    )
    # The actionable match should now rank first.
    assert reranked[0].id == "m"


# ---------------------------------------------------------------------------
# context_gate
# ---------------------------------------------------------------------------

def test_gate_packages_l1_by_default():
    card = ExperienceCard(
        memory_type=MemoryType.EVENT, trigger="NPE", root_cause="None token",
        action_hint="add null check", files_symbols=["auth.py"],
        failure_if_ignored="500 on every request",
    )
    note = _note("m", "auth.py NPE fix", card)
    res = SearchResult(id="m", content="auth.py NPE fix", score=1.0, created_at=note.created_at, memory_type=MemoryType.EVENT)
    gated = gate(None, [res], {"m": note})
    assert len(gated) == 1
    assert gated[0].granularity == "L1"
    assert "trigger: NPE" in gated[0].content
    assert "action: add null check" in gated[0].content


def test_gate_l0_is_one_liner_bounded():
    card = ExperienceCard(
        memory_type=MemoryType.EVENT, trigger="NPE on token",
        action_hint="add null check before accessing token.user in auth.py verify method",
    )
    note = _note("m", "x", card)
    res = SearchResult(id="m", content="x", score=1.0, created_at=note.created_at, memory_type=MemoryType.EVENT)
    params = DomainParams(l0_max_chars=60)
    gated = gate(None, [res], {"m": note}, granularity="L0", params=params)
    assert gated[0].granularity == "L0"
    assert len(gated[0].content) <= 60
    assert "→" in gated[0].content


def test_gate_l2_includes_full_content():
    card = ExperienceCard(memory_type=MemoryType.RECORD, gist="the signature", entities=["foo"])
    note = _note("m", "def foo(x): return x", card, mtype=MemoryType.RECORD)
    res = SearchResult(id="m", content="def foo(x): return x", score=1.0, created_at=note.created_at, memory_type=MemoryType.RECORD)
    gated = gate(None, [res], {"m": note}, granularity="L2")
    assert gated[0].granularity == "L2"
    assert "def foo(x): return x" in gated[0].content


def test_gate_filters_zero_weight_type_keeps_min():
    """A zero-weight type is dropped unless it would leave nothing."""
    card_ev = ExperienceCard(memory_type=MemoryType.EVENT, action_hint="a")
    card_rec = ExperienceCard(memory_type=MemoryType.RECORD, gist="g")
    n_ev = _note("e", "event text", card_ev, mtype=MemoryType.EVENT)
    n_rec = _note("r", "record text", card_rec, mtype=MemoryType.RECORD)
    res = [
        SearchResult(id="e", content="event text", score=1.0, created_at=n_ev.created_at, memory_type=MemoryType.EVENT),
        SearchResult(id="r", content="record text", score=0.5, created_at=n_rec.created_at, memory_type=MemoryType.RECORD),
    ]
    # Query wants only event → record has zero weight.
    weights = {MemoryType.EVENT: 1.0, MemoryType.RECORD: 0.0, MemoryType.PROFILE: 0.0}
    gated = gate(weights, res, {"e": n_ev, "r": n_rec})
    ids = [g.id for g in gated]
    assert "e" in ids
    # record is below median score and zero-weight → dropped.
    assert "r" not in ids


def test_package_legacy_note_without_card():
    """A legacy note with no card falls back to raw content."""
    note = _note("m", "raw content no card")
    res = SearchResult(id="m", content="raw content no card", score=1.0, created_at=note.created_at, memory_type=MemoryType.RECORD)
    g = package(res, None, note.content, "L1")
    assert g.content == "raw content no card"
