"""Domain models: pure data classes for the code memory system.

No IO, no external dependencies. All persistence concerns live in
`infrastructure/`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .enums import MemoryType


def _utcnow_iso() -> str:
    """Current time in UTC, ISO-8601 format with timezone offset."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryNote:
    """In-memory representation of a memory entry.

    `type_specific` carries the type-dependent structured fields:
    - PROFILE: {"pairs": [{"attr": str, "value": str}, ...]}
    - EVENT:   {"trigger": str, "action": str, "outcome": str, "temporal_anchor": str}
    - RECORD:  {"gist": str, "entities": [str, ...], "keywords": [str, ...]}
    Plus a `card` sub-dict (CICL 5 fields) for all types.

    Version chain: `version` increments on merge/refine; `supersedes_id`
    points at the note this one replaces (contradicts); `superseded` marks old
    versions excluded from default search.
    """

    id: str
    user_id: str
    session_id: str
    request_id: str
    content: str
    embedding: list[float]
    memory_type: MemoryType
    type_specific: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utcnow_iso)
    # Version chain
    version: int = 1
    supersedes_id: str | None = None
    superseded: bool = False

    def to_row(self) -> dict[str, Any]:
        """Serialize to a row-shape dict suitable for `infrastructure/sqlite.py`."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "content": self.content,
            "embedding": self.embedding,
            "memory_type": self.memory_type.value,
            "type_specific": self.type_specific,
            "created_at": self.created_at,
            "version": self.version,
            "supersedes_id": self.supersedes_id,
            "superseded": 1 if self.superseded else 0,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "MemoryNote":
        """Inverse of `to_row`. Missing `type_specific` defaults to {}."""
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            request_id=row["request_id"],
            content=row["content"],
            embedding=list(row["embedding"]),
            memory_type=MemoryType(row["memory_type"]),
            type_specific=dict(row.get("type_specific") or {}),
            created_at=row["created_at"],
            version=int(row.get("version") or 1),
            supersedes_id=row.get("supersedes_id"),
            superseded=bool(row.get("superseded") or 0),
        )


@dataclass
class SearchResult:
    """A single retrieval hit returned by the search pipeline."""

    id: str
    content: str
    score: float
    created_at: str
    memory_type: MemoryType


@dataclass
class QueryPlan:
    """Adaptive retrieval plan (LeanMem §3.4)."""

    enhanced_query: str
    type_weights: dict[MemoryType, float]  # may be empty → no type bias
