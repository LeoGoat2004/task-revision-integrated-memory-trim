"""Tests for transcript_parser: raw JSONL + normalized {role,content} shapes."""
from __future__ import annotations

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.domain.transcript_parser import (
    TaskContext,
    TranscriptEvent,
    events_to_text,
    parse_transcript,
)


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "transcript_8c0be6e5.jsonl"


def _load_fixture() -> list[dict]:
    rows = []
    with open(FIXTURE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_parse_raw_transcript_extracts_task_context():
    rows = _load_fixture()
    events, ctx = parse_transcript(rows)
    assert ctx.instance_id == "django__django-16527"
    assert ctx.repo == "django/django"
    assert ctx.language == "python"
    assert ctx.problem_statement  # non-empty
    assert len(ctx.base_commit) == 40


def test_parse_raw_transcript_produces_typed_events():
    rows = _load_fixture()
    events, _ = parse_transcript(rows)
    assert len(events) > 0
    kinds = {e.kind for e in events}
    # Must have at least text + tool_use from the assistant turns.
    assert "text" in kinds
    assert "tool_use" in kinds
    # Indices are monotonic.
    assert [e.index for e in events] == list(range(len(events)))


def test_files_touched_aggregated_from_tool_use():
    rows = _load_fixture()
    events, ctx = parse_transcript(rows)
    # The django fixture reads django/contrib/admin/... files.
    tool_use_events = [e for e in events if e.kind == "tool_use"]
    assert any(e.tool_name == "Read" for e in tool_use_events)
    # files_touched on events should be populated for Read/Edit/Write.
    assert any(e.files_touched for e in tool_use_events)
    # Aggregate flows into TaskContext.
    assert any("django" in f for f in ctx.files_touched)


def test_tool_use_input_preserved():
    rows = _load_fixture()
    events, _ = parse_transcript(rows)
    reads = [e for e in events if e.kind == "tool_use" and e.tool_name == "Read"]
    assert reads
    assert isinstance(reads[0].tool_input, dict)
    # Read tool input has file_path.
    assert any("file_path" in e.tool_input or "path" in e.tool_input for e in reads)


def test_normalized_role_content_shape():
    """The /add wire contract: [{role, content:str, timestamp?}]."""
    messages = [
        {"role": "user", "content": "Fix the NPE in auth.py when token is None.", "timestamp": 1700000000},
        {"role": "assistant", "content": "I'll read auth.py and add a null check."},
    ]
    events, ctx = parse_transcript(messages)
    assert len(events) == 2
    assert events[0].role == "user" and events[0].kind == "text"
    assert events[0].ts == "1700000000"
    assert events[1].role == "assistant"
    # No instance_id in normalized payload → problem_statement falls back.
    assert "NPE" in ctx.problem_statement


def test_normalized_with_serialized_tool_use():
    messages = [
        {"role": "user", "content": "Fix bug in util.py"},
        {"role": "assistant", "content": "Reading util.py\n<tool_use name=\"Read\">file_path: util.py\n</tool_use>"},
    ]
    events, _ = parse_transcript(messages)
    kinds = [e.kind for e in events]
    assert "tool_use" in kinds
    tu = next(e for e in events if e.kind == "tool_use")
    assert tu.tool_name == "Read"
    assert tu.tool_input.get("file_path") == "util.py"
    assert "util.py" in tu.files_touched


def test_events_to_text_bounded_and_informative():
    rows = _load_fixture()
    events, _ = parse_transcript(rows)
    text = events_to_text(events, tool_result_cap=200)
    # Contains tool labels (localization signal) and user task.
    assert "tool:" in text or "tool_result" in text
    # Bounded: not the full raw transcript size.
    assert len(text) < 20000


def test_mixed_input_tolerated():
    """A mix of raw rows and normalized dicts must not crash."""
    rows = _load_fixture()[:2]
    rows.append({"role": "assistant", "content": "appended normalized note"})
    events, _ = parse_transcript(rows)
    assert len(events) >= 2
    assert events[-1].kind == "text"


def test_empty_input():
    events, ctx = parse_transcript([])
    assert events == []
    assert ctx.instance_id == ""
