"""Domain enums for the code memory system.

MemoryType is borrowed from LeanMem (arXiv:2608.03463, 2026-08-04).
"""
from __future__ import annotations

from enum import Enum


class MemoryType(str, Enum):
    """Three memory types, each with its own compression strategy.

    - PROFILE: stable, long-lived project/codebase facts. Compressed into
      (attribute, value) pairs (no LLM call; heuristic regex per
      LeanMem §3.2). Reused across many queries.
    - EVENT: dynamic, time/state-bound activities (debugging, fix, deploy).
      Compressed into (trigger, action, outcome, temporal_anchor). Subject to
      Selective Memory Evolution (LeanMem §3.3).
    - RECORD: detail-dense, source-grounded artifacts (function signatures,
      API usage, error messages, configs). Compressed into (gist, entities,
      keywords). Immutable once stored.
    """

    PROFILE = "profile"
    EVENT = "event"
    RECORD = "record"


class QueryIntent(str, Enum):
    """Query intent classifiers, used to dispatch adaptive retrieval."""

    PROFILE = "profile"
    EVENT = "event"
    RECORD = "record"
