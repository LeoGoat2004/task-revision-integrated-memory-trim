"""SQLite persistence layer for the code memory system.

Schema
------
memories
  id              TEXT PRIMARY KEY
  user_id         TEXT NOT NULL (isolation boundary)
  session_id      TEXT NOT NULL
  request_id      TEXT NOT NULL
  content         TEXT NOT NULL
  embedding       BLOB NOT NULL (float32 LE)
  memory_type     TEXT NOT NULL (profile / event / record)
  type_specific   TEXT NOT NULL (JSON-serialized structured fields)
  created_at      TEXT NOT NULL (ISO-8601 UTC)

Indexes
  - idx_memories_user        (user_id only)
  - idx_memories_user_type   (user_id, memory_type) — boosts type-filtered lookup

Migration
  - On init, PRAGMA table_info is inspected and missing columns are added
    via ALTER TABLE. This makes schema upgrades non-destructive.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from ..config import get_settings
from ..domain.enums import MemoryType
from ..domain.models import MemoryNote


_local = threading.local()


def _connect() -> sqlite3.Connection:
    settings = get_settings()
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_conn() -> sqlite3.Connection:
    """Return the per-thread connection (creates one on first use)."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _connect()
        _local.conn = conn
    return conn


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------

# The DDL below is the canonical schema. The init / migration code
# tolerates older databases that lack the `memory_type` / `type_specific`
# columns added later.

_DDL = """
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    request_id      TEXT NOT NULL,
    content         TEXT NOT NULL,
    embedding       BLOB NOT NULL,
    memory_type     TEXT NOT NULL DEFAULT 'record',
    type_specific   TEXT NOT NULL DEFAULT '{{}}',
    created_at      TEXT NOT NULL,
    version         INTEGER NOT NULL DEFAULT 1,
    supersedes_id   TEXT,
    superseded      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_memories_user        ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_user_type   ON memories(user_id, memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_superseded  ON memories(user_id, superseded);

CREATE TABLE IF NOT EXISTS file_symbols (
    user_id         TEXT NOT NULL,
    file            TEXT NOT NULL,
    symbols         TEXT NOT NULL DEFAULT '[]',
    lang            TEXT NOT NULL DEFAULT '',
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (user_id, file)
);
"""

# Pairs of (column_name, column_ddl) used by the migration step.
# memory_type + type_specific were added to support typed memories; the
# version + supersedes_id + superseded columns back the version chain for
# revision.ingest.
_EXPECTED_COLUMNS: dict[str, str] = {
    "memory_type": "TEXT NOT NULL DEFAULT 'record'",
    "type_specific": "TEXT NOT NULL DEFAULT '{}'",
    "version": "INTEGER NOT NULL DEFAULT 1",
    "supersedes_id": "TEXT",
    "superseded": "INTEGER NOT NULL DEFAULT 0",
}


def init_db() -> None:
    """Create tables (if missing) and add new columns to older databases."""
    conn = get_conn()
    conn.executescript(_DDL)
    conn.commit()
    migrate_if_needed(conn)


def migrate_if_needed(conn: sqlite3.Connection) -> None:
    """Add missing columns introduced after the initial schema."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
    for col, ddl in _EXPECTED_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {ddl}")
    # Version-chain index — create if missing (safe to retry).
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_superseded "
        "ON memories(user_id, superseded)"
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Read / write helpers
# ---------------------------------------------------------------------------


def _row_to_note(row: sqlite3.Row) -> MemoryNote:
    """Convert a sqlite3.Row to a MemoryNote, decoding the BLOB embedding."""
    emb_blob = row["embedding"]
    embedding = list(np.frombuffer(emb_blob, dtype=np.float32)) if emb_blob else []
    type_specific_raw = row["type_specific"] or "{}"
    try:
        type_specific = json.loads(type_specific_raw)
        if not isinstance(type_specific, dict):
            type_specific = {}
    except json.JSONDecodeError:
        type_specific = {}
    raw_type = row["memory_type"]
    try:
        memory_type = MemoryType(raw_type)
    except ValueError:
        # Defensive: fall back to RECORD for unknown / legacy values.
        memory_type = MemoryType.RECORD
    return MemoryNote(
        id=row["id"],
        user_id=row["user_id"],
        session_id=row["session_id"],
        request_id=row["request_id"],
        content=row["content"],
        embedding=embedding,
        memory_type=memory_type,
        type_specific=type_specific,
        created_at=row["created_at"],
        version=int(row["version"] if "version" in row.keys() else 1 or 1),
        supersedes_id=row["supersedes_id"] if "supersedes_id" in row.keys() else None,
        superseded=bool(row["superseded"] if "superseded" in row.keys() else 0),
    )


# Column list shared by INSERT and SELECT so they never drift.
_MEM_COLS = (
    "id, user_id, session_id, request_id, content, embedding, "
    "memory_type, type_specific, created_at, version, supersedes_id, superseded"
)


def insert_memory(note: MemoryNote) -> None:
    """Insert a memory note. Uses `INSERT OR REPLACE` for idempotency."""
    conn = get_conn()
    emb_bytes = np.asarray(note.embedding, dtype=np.float32).tobytes()
    conn.execute(
        """
        INSERT OR REPLACE INTO memories(
            id, user_id, session_id, request_id, content,
            embedding, memory_type, type_specific, created_at,
            version, supersedes_id, superseded
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            note.id,
            note.user_id,
            note.session_id,
            note.request_id,
            note.content,
            emb_bytes,
            note.memory_type.value,
            json.dumps(note.type_specific, ensure_ascii=False),
            note.created_at,
            note.version,
            note.supersedes_id,
            1 if note.superseded else 0,
        ),
    )
    conn.commit()


def update_note(note: MemoryNote) -> None:
    """Update an existing note's mutable fields (content, type_specific, version).

    Used by revision.ingest MERGE: the merged card + content replace the old
    row, version is bumped. Embedding is NOT re-embedded here (the service
    layer re-embeds merged content before calling this).
    """
    conn = get_conn()
    conn.execute(
        """
        UPDATE memories SET
            content = ?, type_specific = ?, memory_type = ?,
            version = ?, created_at = ?
        WHERE id = ?
        """,
        (
            note.content,
            json.dumps(note.type_specific, ensure_ascii=False),
            note.memory_type.value,
            note.version,
            note.created_at,
            note.id,
        ),
    )
    conn.commit()


def mark_superseded(old_id: str, new_id: str | None) -> None:
    """Flag `old_id` as superseded by `new_id` (excluded from default search)."""
    conn = get_conn()
    conn.execute(
        "UPDATE memories SET superseded = 1, supersedes_id = ? WHERE id = ?",
        (new_id, old_id),
    )
    conn.commit()


def fetch_memories_by_user(
    user_id: str, *, include_superseded: bool = False
) -> list[MemoryNote]:
    """Return all active memories for a user, ordered by created_at descending.

    By default excludes superseded (old-version) notes so search sees only the
    latest version of each memory.
    """
    conn = get_conn()
    sql = f"SELECT {_MEM_COLS} FROM memories WHERE user_id = ?"
    params: list = [user_id]
    if not include_superseded:
        sql += " AND superseded = 0"
    sql += " ORDER BY created_at DESC"
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_note(r) for r in rows]


def fetch_memories_by_type(
    user_id: str, memory_type: str, limit: int | None = None
) -> list[MemoryNote]:
    """Return memories for a user filtered by type (excludes superseded)."""
    conn = get_conn()
    sql = (
        f"SELECT {_MEM_COLS} FROM memories "
        "WHERE user_id = ? AND memory_type = ? AND superseded = 0 "
        "ORDER BY created_at DESC"
    )
    if limit is not None:
        sql += " LIMIT ?"
        rows = conn.execute(sql, (user_id, memory_type, limit)).fetchall()
    else:
        rows = conn.execute(sql, (user_id, memory_type)).fetchall()
    return [_row_to_note(r) for r in rows]


def fetch_recent_memories(
    user_id: str, limit: int = 50
) -> list[MemoryNote]:
    """Return the most recent `limit` active memories for a user."""
    conn = get_conn()
    rows = conn.execute(
        f"SELECT {_MEM_COLS} FROM memories "
        "WHERE user_id = ? AND superseded = 0 ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return [_row_to_note(r) for r in rows]


def count_memories() -> int:
    """Return the total number of memories in the database."""
    conn = get_conn()
    row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
    return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# file_symbols store (symbol graph)
# ---------------------------------------------------------------------------

def upsert_file_symbols(
    user_id: str, file: str, symbols: list[dict], lang: str
) -> None:
    """Insert-or-replace a file's extracted symbols for a user.

    `symbols` is a list of SymbolRef-shaped dicts ({name, kind, line, ...}).
    Stored as JSON so the domain layer can rebuild the graph without IO types.
    """
    conn = get_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO file_symbols(user_id, file, symbols, lang, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            user_id,
            file,
            json.dumps(symbols, ensure_ascii=False),
            lang,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def fetch_file_symbols(user_id: str) -> list[dict]:
    """Return all stored symbol refs for a user as a flat list of dicts.

    Each dict carries `file`, `name`, `kind`, `lang`, `node_type` — the shape
    `domain.symbol_graph.build_graph` consumes.
    """
    conn = get_conn()
    rows = conn.execute(
        "SELECT file, symbols, lang FROM file_symbols WHERE user_id = ?",
        (user_id,),
    ).fetchall()
    out: list[dict] = []
    for row in rows:
        file = row["file"]
        lang = row["lang"]
        try:
            syms = json.loads(row["symbols"] or "[]")
        except json.JSONDecodeError:
            syms = []
        for s in syms:
            if not isinstance(s, dict):
                continue
            out.append({
                "file": file,
                "name": s.get("name", ""),
                "kind": s.get("kind", ""),
                "lang": lang,
                "node_type": s.get("node_type", ""),
                "line": s.get("line", 0),
            })
    return out


def reset_db() -> None:
    """Wipe the database (tests only)."""
    settings = get_settings()
    Path(settings.db_path).unlink(missing_ok=True)
    _local.conn = None
