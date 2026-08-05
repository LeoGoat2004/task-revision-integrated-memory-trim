"""Tests for task-boundary segmentation."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.domain.chunker import segment_events
from app.domain.params import DomainParams
from app.domain.transcript_parser import TranscriptEvent


def _ev(role="assistant", kind="text", text="", tool_name="", ts="", tool_result=""):
    return TranscriptEvent(
        role=role, kind=kind, text=text, tool_name=tool_name, ts=ts,
        tool_result=tool_result,
    )


def test_empty_returns_empty():
    assert segment_events([]) == []


def test_single_segment_when_small():
    events = [
        _ev("user", "text", "Fix the bug"),
        _ev("assistant", "text", "I will read the file"),
        _ev("assistant", "tool_use", tool_name="Read"),
    ]
    segs = segment_events(events)
    assert len(segs) == 1
    assert len(segs[0]) == 3


def test_todowrite_starts_new_segment():
    events = [
        _ev("assistant", "text", "step 1"),
        _ev("assistant", "tool_use", tool_name="Read"),
        _ev("assistant", "tool_use", tool_name="TodoWrite"),  # boundary
        _ev("assistant", "text", "step 2 after replan"),
        _ev("assistant", "tool_use", tool_name="Edit"),
    ]
    segs = segment_events(events)
    assert len(segs) == 2
    # TodoWrite opens the second segment.
    assert segs[1][0].tool_name == "TodoWrite"
    assert segs[0][-1].tool_name == "Read"


def test_idle_gap_starts_new_segment():
    events = [
        _ev("assistant", "text", "working", ts="2025-12-09T20:00:00Z"),
        _ev("assistant", "tool_use", tool_name="Read", ts="2025-12-09T20:00:30Z"),
        # 20 minute gap
        _ev("user", "text", "resumed", ts="2025-12-09T20:20:00Z"),
    ]
    segs = segment_events(events)
    assert len(segs) == 2
    assert segs[1][0].role == "user"


def test_max_messages_cap():
    params = DomainParams(chunk_max_messages=3, chunk_boundary_on_todo=True)
    events = [_ev("assistant", "text", f"line {i}") for i in range(7)]
    segs = segment_events(events, params)
    assert all(len(s) <= 3 for s in segs)
    assert sum(len(s) for s in segs) == 7


def test_max_words_cap():
    params = DomainParams(chunk_max_words=10, chunk_boundary_on_todo=False)
    big = "word " * 8  # 8 words
    events = [_ev("assistant", "text", big), _ev("assistant", "text", big), _ev("assistant", "text", big)]
    segs = segment_events(events, params)
    # Each segment must not exceed the word cap by more than one event.
    for s in segs:
        assert sum(len(e.text.split()) for e in s) <= 16  # cap + one event overflow


def test_tool_use_and_result_not_split_across_boundary():
    """A tool_result immediately following its tool_use stays in the same segment."""
    events = [
        _ev("assistant", "text", "x" * 5, ts="2025-12-09T20:00:00Z"),
        _ev("assistant", "tool_use", tool_name="Read", ts="2025-12-09T20:00:10Z"),
        # gap would trigger boundary, but tool_result should stick to its tool_use
        _ev("user", "tool_result", tool_result="file contents", ts="2025-12-09T20:11:00Z"),
    ]
    segs = segment_events(events)
    # The tool_use and tool_result land in the same segment.
    last_seg = segs[-1]
    kinds = [e.kind for e in last_seg]
    assert "tool_use" in kinds and "tool_result" in kinds
