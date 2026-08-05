"""Tests for ExperienceCard schema + heuristic extractor."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.domain.enums import MemoryType
from app.domain.experience_card import (
    ExperienceCard,
    classify_heuristic,
    heuristic_card,
)


def test_card_roundtrip_record():
    card = ExperienceCard(
        memory_type=MemoryType.RECORD,
        trigger="AttributeError on None token",
        root_cause="token not checked before .user access",
        action_hint="guard with `if token is not None` before attribute access",
        files_symbols=["auth.py", "AuthService.verify"],
        failure_if_ignored="500 on every unauthenticated request",
        scope="myapp/python#auth",
        temporal_anchor="v1.2.3",
        gist="Guard None token in AuthService.verify to avoid AttributeError",
        entities=["AuthService", "verify", "AttributeError"],
        keywords=["auth", "token", "None", "guard"],
    )
    ts = card.to_type_specific()
    assert ts["card"]["root_cause"] == "token not checked before .user access"
    assert ts["gist"] == card.gist
    back = ExperienceCard.from_type_specific(card.memory_type, ts)
    assert back.root_cause == card.root_cause
    assert back.files_symbols == card.files_symbols
    assert back.memory_type == MemoryType.RECORD


def test_card_backward_compat_v1_event():
    """Legacy event shape (trigger/action/outcome) is recoverable."""
    v1_ts = {
        "trigger": "NPE on login",
        "action": "added null check",
        "outcome": "fixed in v1.2",
        "temporal_anchor": "v1.2",
    }
    card = ExperienceCard.from_type_specific(MemoryType.EVENT, v1_ts)
    assert card.trigger == "NPE on login"
    assert card.action_hint == "added null check"
    assert card.root_cause == "fixed in v1.2"
    assert card.temporal_anchor == "v1.2"


def test_card_backward_compat_v1_record():
    v1_ts = {"gist": "g", "entities": ["a"], "keywords": ["k"]}
    card = ExperienceCard.from_type_specific(MemoryType.RECORD, v1_ts)
    assert card.gist == "g"
    assert card.entities == ["a"]


def test_classify_heuristic():
    assert classify_heuristic("Fixed the NPE in auth.py") == MemoryType.EVENT
    assert classify_heuristic("Project uses Python 3.12 with FastAPI") == MemoryType.PROFILE
    assert classify_heuristic("def foo(x: int) -> str in util.py") == MemoryType.RECORD


def test_heuristic_card_event():
    text = "Fixed the AttributeError in auth.py by adding a null check for the token before accessing token.user."
    card = heuristic_card(text, repo="myapp", language="python", files_touched={"auth.py"})
    assert card.memory_type == MemoryType.EVENT
    assert "auth.py" in card.files_symbols
    assert card.scope == "myapp/python"
    assert card.trigger  # populated from error hint
    assert card.action_hint


def test_heuristic_card_record_extracts_entities():
    text = "AuthService.verify(token) raises AttributeError when token is None. See util.py:42."
    card = heuristic_card(text, repo="myapp", language="python")
    assert card.memory_type == MemoryType.RECORD
    assert card.entities  # AuthService.verify or similar
    assert card.keywords


def test_heuristic_card_never_fabricates_root_cause():
    """Conservative: record type without explicit cause leaves root_cause empty."""
    text = "The config file config.yaml has timeout=30."
    card = heuristic_card(text)
    assert card.root_cause == ""


def test_to_llm_dict_shape():
    card = ExperienceCard(memory_type=MemoryType.EVENT, trigger="t")
    d = card.to_llm_dict()
    assert d["memory_type"] == "event"
    assert "files_symbols" in d and "failure_if_ignored" in d
