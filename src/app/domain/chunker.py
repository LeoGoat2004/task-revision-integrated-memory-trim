"""Task-boundary segmentation of the transcript event stream.

Naive segmentation split raw messages by fixed count (20 msgs / 2000 words).
Real Claude Code transcripts have natural boundaries — TodoWrite (agent
re-plans), long idle gaps, and tool-call clusters around one sub-task — that
yield far better ExperienceCards when respected. This module segments
`TranscriptEvent`s on those boundaries, with hard caps from `DomainParams`
as a safety net.

Rules (in priority order):
  1. A TodoWrite tool_use starts a new segment (the agent explicitly re-planned).
  2. A timestamp gap > `IDLE_GAP_SECS` between consecutive events starts a new
     segment (the human/agent paused).
  3. A segment is flushed when it hits `chunk_max_messages` or `chunk_max_words`.
  4. A tool_use and its immediately-following tool_result are never split.

Pure function over events + params; no IO.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from .params import DomainParams, DEFAULT_PARAMS

if TYPE_CHECKING:
    from .transcript_parser import TranscriptEvent
    from ..api.schemas import AddMessage


# A gap larger than this (seconds) between consecutive events implies a pause.
IDLE_GAP_SECS: int = 600


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _event_weight(ev: "TranscriptEvent") -> int:
    """Approximate word cost of an event (text + truncated tool_result)."""
    parts = [ev.text, ev.tool_result]
    if ev.kind == "tool_use":
        # Count key input values as words (localization signal).
        for v in ev.tool_input.values():
            if isinstance(v, str):
                parts.append(v)
    return sum(_word_count(p) for p in parts)


def _parse_ts(ts: str) -> float | None:
    """Parse an ISO-8601 or unix-ms timestamp to seconds-since-epoch."""
    if not ts:
        return None
    # Unix milliseconds (the wire contract sends ints as str).
    if ts.isdigit():
        try:
            v = int(ts)
            # Heuristic: ms if very large, else seconds.
            return v / 1000.0 if v > 1e12 else float(v)
        except ValueError:
            return None
    # ISO-8601 (Claude Code transcripts use 2025-12-09T20:18:28.901Z).
    from datetime import datetime
    try:
        s = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return None


def segment_events(
    events: Sequence["TranscriptEvent"],
    params: DomainParams = DEFAULT_PARAMS,
) -> list[list["TranscriptEvent"]]:
    """Segment a transcript event stream by task boundaries.

    Returns a list of segments, each a list of `TranscriptEvent` in original
    order. Empty input → empty list (caller decides whether to skip).
    """
    if not events:
        return []

    segments: list[list["TranscriptEvent"]] = []
    cur: list["TranscriptEvent"] = []
    cur_words = 0
    prev_ts: float | None = None

    def _flush() -> None:
        nonlocal cur, cur_words
        if cur:
            segments.append(cur)
        cur = []
        cur_words = 0

    for ev in events:
        # --- Boundary detection (only flush if cur is non-empty) ---
        if cur:
            # Never split a tool_use from its immediately-following tool_result:
            # a tool_use without its result yields a half-story card.
            keep_with_prev = (
                ev.kind == "tool_result"
                and cur[-1].kind == "tool_use"
            )
            boundary = False
            if keep_with_prev:
                boundary = False
            elif params.chunk_boundary_on_todo and ev.kind == "tool_use" and ev.tool_name == "TodoWrite":
                boundary = True
            else:
                ts = _parse_ts(ev.ts)
                if ts is not None and prev_ts is not None and (ts - prev_ts) > IDLE_GAP_SECS:
                    boundary = True
                elif len(cur) >= params.chunk_max_messages:
                    boundary = True
                elif cur_words + _event_weight(ev) > params.chunk_max_words:
                    boundary = True
            if boundary:
                _flush()

        cur.append(ev)
        cur_words += _event_weight(ev)

        ts = _parse_ts(ev.ts)
        if ts is not None:
            prev_ts = ts

    _flush()
    return segments


# ---------------------------------------------------------------------------
# Backward-compatible adapter for the legacy wire contract ({role, content}).
# Used by tests and any caller that hasn't migrated to TranscriptEvent yet.
# ---------------------------------------------------------------------------

def total_word_count(messages: Sequence["AddMessage"]) -> int:
    return sum(_word_count(getattr(m, "content", "") or "") for m in messages)


def should_chunk(messages: list["AddMessage"], params: DomainParams = DEFAULT_PARAMS) -> bool:
    if len(messages) > params.chunk_max_messages:
        return True
    return total_word_count(messages) > params.chunk_max_words


def chunk_messages(
    messages: list["AddMessage"], params: DomainParams = DEFAULT_PARAMS
) -> list[list["AddMessage"]]:
    """Legacy fixed-cap chunker over plain {role, content} messages.

    Kept so the wire contract still works when callers send normalized messages
    without going through the transcript parser. The new pipeline prefers
    `segment_events` over `TranscriptEvent`s.
    """
    if not messages:
        return []
    if not should_chunk(messages, params):
        return [list(messages)]

    chunks: list[list["AddMessage"]] = []
    cur: list["AddMessage"] = []
    cur_words = 0
    for msg in messages:
        words = _word_count(getattr(msg, "content", "") or "")
        over_count = bool(cur) and len(cur) >= params.chunk_max_messages
        over_words = bool(cur) and cur_words + words > params.chunk_max_words
        if over_count or over_words:
            chunks.append(cur)
            cur, cur_words = [], 0
        cur.append(msg)
        cur_words += words
    if cur:
        chunks.append(cur)
    return chunks or [list(messages)]
