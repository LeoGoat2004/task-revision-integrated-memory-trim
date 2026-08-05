"""Transcript parser: normalize heterogeneous /add inputs into a unified event
stream.

The /add contract accepts `messages: [{role, content, timestamp?}]`, but the
*real* SWEContextBench "Past Experience" data is a Claude Code session
transcript (JSONL) where:
  - `user` rows carry `message.content` as a string (the task) OR a list of
    `tool_result` blocks.
  - `assistant` rows carry `message.content` as a list of `text` / `thinking` /
    `tool_use` blocks.
  - rich metadata: cwd, gitBranch, sessionId, timestamp, todos.

This parser tolerates **both** shapes:
  1. Raw transcript rows (dicts with `type`/`message`).
  2. Normalized `{role, content:str}` (the wire contract; content may already
     embed serialized tool calls as text).

Output: a list of `TranscriptEvent` plus a parsed `TaskContext` (instance_id,
repo, problem_statement, files_touched) extracted from the first user message.
The event stream is what the chunker segments and the ExperienceCard extractor
consumes — so downstream code never re-parses raw JSON.

Pure function: no IO, no LLM. Fully unit-testable with fixtures.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TranscriptEvent:
    """A single normalized event in a coding-session transcript.

    `kind` discriminates the payload:
      - "text": assistant free text or user task text → `text` populated.
      - "thinking": assistant reasoning → `text` populated (kept for context,
        down-weighted in extraction).
      - "tool_use": assistant invoked a tool → `tool_name` + `tool_input`.
      - "tool_result": user-side result of a tool call → `tool_result`.
      - "snapshot": file-history snapshot (rarely useful, kept for completeness).
    """

    role: str  # "user" | "assistant"
    kind: str  # "text" | "thinking" | "tool_use" | "tool_result" | "snapshot"
    text: str = ""
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_result: str = ""
    files_touched: list[str] = field(default_factory=list)
    cwd: str = ""
    git_branch: str = ""
    ts: str = ""
    todos: list[Any] = field(default_factory=list)
    # Monotonic index in the event stream (set by parse_transcript).
    index: int = 0


@dataclass
class TaskContext:
    """Metadata extracted from the transcript's first user message.

    SWEContextBench transcripts embed the task as:
        Fix this bug to solve the issue based on manual.yaml:
          instance_id: django__django-12915
          repo: django/django
          base_commit: <sha>
          problem_statement: "..."
    We parse these fields to seed the ExperienceCard `scope` / `temporal_anchor`.
    """

    instance_id: str = ""
    repo: str = ""
    base_commit: str = ""
    problem_statement: str = ""
    # Aggregate of every file referenced by Read/Edit/Write/Grep tool calls.
    files_touched: set[str] = field(default_factory=set)
    # The primary language inferred from repo / file extensions.
    language: str = ""


# ---------------------------------------------------------------------------
# Task-context extraction
# ---------------------------------------------------------------------------

_INSTANCE_ID_RE = re.compile(r"instance_id:\s*([A-Za-z0-9_.\-/]+)")
_REPO_RE = re.compile(r"(?m)^\s*repo:\s*([A-Za-z0-9_.\-/]+)\s*$")
_BASE_COMMIT_RE = re.compile(r"base_commit:\s*([0-9a-fA-F]+)")
# problem_statement is usually a YAML-quoted or plain multi-line block after
# the `problem_statement:` key; we capture until the next top-level yaml key or
# the `Description` header that SWE-bench issues include.
_PROBLEM_STMT_RE = re.compile(
    r"problem_statement:\s*[\"']?(.*?)(?=\n[A-Za-z_]+:|\nDescription\b|\Z)",
    re.DOTALL,
)

# repo → primary language mapping for the repos covered by SWEContextBench.
_REPO_LANG: dict[str, str] = {
    "django/django": "python",
    "astropy/astropy": "python",
    "matplotlib/matplotlib": "python",
    "mwaskom/seaborn": "python",
    "pallets/flask": "python",
    "psf/requests": "python",
    "pydata/xarray": "python",
    "pylint-dev/pylint": "python",
    "pytest-dev/pytest": "python",
    "scikit-learn/scikit-learn": "python",
    "sphinx-doc/sphinx": "python",
    "sympy/sympy": "python",
}

_EXT_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".rs": "rust",
}


def infer_language(repo: str, files: Iterable[str]) -> str:
    """Infer the primary language from repo name, falling back to file exts."""
    lang = _REPO_LANG.get(repo)
    if lang:
        return lang
    counts: dict[str, int] = {}
    for f in files:
        dot = f.rfind(".")
        if dot == -1:
            continue
        ext = f[dot:].lower()
        l = _EXT_LANG.get(ext)
        if l:
            counts[l] = counts.get(l, 0) + 1
    if counts:
        return max(counts, key=counts.get)
    return ""


def _extract_task_context(first_user_text: str) -> TaskContext:
    """Parse the embedded YAML-ish task header from the first user message."""
    ctx = TaskContext()
    if not first_user_text:
        return ctx
    m = _INSTANCE_ID_RE.search(first_user_text)
    if m:
        ctx.instance_id = m.group(1).strip()
    m = _REPO_RE.search(first_user_text)
    if m:
        ctx.repo = m.group(1).strip()
    m = _BASE_COMMIT_RE.search(first_user_text)
    if m:
        ctx.base_commit = m.group(1).strip()
    m = _PROBLEM_STMT_RE.search(first_user_text)
    if m:
        stmt = m.group(1).strip().strip("\"'")
        # Collapse excessive whitespace but keep newlines for readability.
        ctx.problem_statement = stmt
    return ctx


# ---------------------------------------------------------------------------
# Files-touched extraction from tool inputs
# ---------------------------------------------------------------------------

# Tools whose inputs reference a primary file path.
_FILE_PATH_KEYS = ("file_path", "path", "filename", "file")


def _extract_files_from_tool_input(tool_name: str, tool_input: Any) -> list[str]:
    """Pull file paths out of a tool_use input dict.

    Handles Read/Edit/Write (file_path), Grep/Glob (path), Bash (command text
    is scanned for quoted paths). Returns de-duplicated, non-empty paths.
    """
    out: list[str] = []
    if not isinstance(tool_input, dict):
        return out
    for key in _FILE_PATH_KEYS:
        val = tool_input.get(key)
        if isinstance(val, str) and val.strip():
            out.append(val.strip())
    # Bash commands: scan for file-like tokens.
    if tool_name == "Bash":
        cmd = tool_input.get("command") or ""
        if isinstance(cmd, str):
            # Match quoted paths or bare path-like tokens with a known ext.
            for m in re.finditer(r"[\"']([^\"']+\.[A-Za-z]{1,4})[\"']", cmd):
                out.append(m.group(1))
    return list(dict.fromkeys(out))  # de-dup preserving order


# ---------------------------------------------------------------------------
# Row-level parsing (raw transcript shape)
# ---------------------------------------------------------------------------

def _parse_content_blocks(
    blocks: list[Any],
    role: str,
    common: dict[str, Any],
) -> list[TranscriptEvent]:
    """Expand a `message.content` block list into TranscriptEvents."""
    events: list[TranscriptEvent] = []
    for blk in blocks:
        if not isinstance(blk, dict):
            continue
        btype = blk.get("type", "")
        if btype == "text":
            events.append(TranscriptEvent(
                role=role, kind="text", text=str(blk.get("text", "")), **common
            ))
        elif btype == "thinking":
            events.append(TranscriptEvent(
                role=role, kind="thinking", text=str(blk.get("thinking", "")), **common
            ))
        elif btype == "tool_use":
            tool_name = str(blk.get("name", ""))
            tool_input = blk.get("input") or {}
            if not isinstance(tool_input, dict):
                tool_input = {"value": tool_input}
            files = _extract_files_from_tool_input(tool_name, tool_input)
            events.append(TranscriptEvent(
                role=role, kind="tool_use",
                tool_name=tool_name, tool_input=tool_input,
                files_touched=files, **common,
            ))
        elif btype == "tool_result":
            # tool_result content may be str or list of text blocks.
            rc = blk.get("content")
            if isinstance(rc, list):
                parts = [
                    b.get("text", "") for b in rc if isinstance(b, dict) and b.get("type") == "text"
                ]
                text = "\n".join(p for p in parts if isinstance(p, str))
            else:
                text = str(rc or "")
            events.append(TranscriptEvent(
                role=role, kind="tool_result", tool_result=text, **common
            ))
    return events


def _common_meta(row: dict[str, Any]) -> dict[str, Any]:
    """Extract metadata shared by all events in a row."""
    msg = row.get("message") if isinstance(row.get("message"), dict) else {}
    return {
        "cwd": str(row.get("cwd", "") or msg.get("cwd", "") or ""),
        "git_branch": str(row.get("gitBranch", "") or ""),
        "ts": str(row.get("timestamp", "") or ""),
        "todos": list(row.get("todos") or msg.get("todos") or []),
    }


def _row_to_events(row: Any) -> list[TranscriptEvent]:
    """Convert one raw transcript row into zero or more TranscriptEvents."""
    if not isinstance(row, dict):
        return []
    rtype = row.get("type", "")
    common = _common_meta(row)
    if rtype in ("file-history-snapshot", "summary"):
        return [TranscriptEvent(role="assistant", kind="snapshot", **common)]
    if rtype not in ("user", "assistant"):
        return []
    msg = row.get("message")
    if not isinstance(msg, dict):
        return []
    content = msg.get("content")
    if isinstance(content, list):
        return _parse_content_blocks(content, rtype, common)
    if isinstance(content, str):
        # User task text or assistant plain text.
        kind = "text"
        return [TranscriptEvent(role=rtype, kind=kind, text=content, **common)]
    return []


# ---------------------------------------------------------------------------
# Normalized {role, content} shape (the /add wire contract)
# ---------------------------------------------------------------------------

# Heuristic: detect serialized tool calls like "<tool_use name=Read>" or
# "Tool: Read" or fenced ```tool blocks in a normalized content string.
_TOOL_TAG_RE = re.compile(
    r"<tool_use\s+name=[\"']?([A-Za-z_]+)[\"']?\s*>(.*?)</tool_use>",
    re.DOTALL,
)


def _parse_normalized_message(role: str, content: str, ts: str) -> list[TranscriptEvent]:
    """Parse a normalized {role, content:str} message into events.

    Most platform payloads send plain text; we still try to recover tool_use
    blocks if they were serialized inline, so the chunker can segment on tool
    boundaries.
    """
    if not content:
        return []
    events: list[TranscriptEvent] = []
    common = {"cwd": "", "git_branch": "", "ts": ts, "todos": []}
    last_end = 0
    for m in _TOOL_TAG_RE.finditer(content):
        if m.start() > last_end:
            pre = content[last_end:m.start()].strip()
            if pre:
                events.append(TranscriptEvent(
                    role=role, kind="text", text=pre, **common
                ))
        tool_name = m.group(1)
        body = m.group(2)
        # Best-effort: treat lines `key: value` as the tool input.
        tool_input: dict[str, Any] = {}
        for line in body.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                k = k.strip()
                if k:
                    tool_input[k] = v.strip()
        files = _extract_files_from_tool_input(tool_name, tool_input)
        events.append(TranscriptEvent(
            role=role, kind="tool_use",
            tool_name=tool_name, tool_input=tool_input,
            files_touched=files, **common,
        ))
        last_end = m.end()
    tail = content[last_end:].strip()
    if tail:
        events.append(TranscriptEvent(role=role, kind="text", text=tail, **common))
    if not events:
        events.append(TranscriptEvent(role=role, kind="text", text=content, **common))
    return events


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _is_raw_transcript_row(obj: Any) -> bool:
    """Heuristic: a dict with a `type` of user/assistant is a raw row."""
    return isinstance(obj, dict) and obj.get("type") in ("user", "assistant", "file-history-snapshot", "summary")


def parse_transcript(messages: list[Any]) -> tuple[list[TranscriptEvent], TaskContext]:
    """Parse heterogeneous /add input into a unified event stream + task context.

    Args:
        messages: either raw transcript rows (dicts with `type`/`message`) or
            normalized `{role, content, timestamp?}` dicts. A mix is tolerated.

    Returns:
        (events, task_context). `events` is a flat, index-ordered list of
        `TranscriptEvent`. `task_context` carries instance_id/repo/problem_statement
        and the aggregate files_touched across the whole transcript.
    """
    events: list[TranscriptEvent] = []
    first_user_text = ""

    for msg in messages or []:
        if _is_raw_transcript_row(msg):
            row_events = _row_to_events(msg)
        elif isinstance(msg, dict) and "role" in msg:
            role = str(msg.get("role", "user"))
            content = msg.get("content", "")
            ts = str(msg.get("timestamp", "") or "")
            if isinstance(content, list):
                # Some normalized payloads still use block lists.
                row_events = _parse_content_blocks(content, role, {"cwd": "", "git_branch": "", "ts": ts, "todos": []})
            else:
                row_events = _parse_normalized_message(role, str(content or ""), ts)
        else:
            row_events = []

        for ev in row_events:
            if not first_user_text and ev.role == "user" and ev.kind == "text":
                first_user_text = ev.text
            events.append(ev)

    # Assign monotonic indices.
    for i, ev in enumerate(events):
        ev.index = i

    ctx = _extract_task_context(first_user_text)
    # If no instance_id was found, treat the whole first user text as the
    # problem statement (generic non-SWEContextBench payload).
    if not ctx.problem_statement and first_user_text:
        ctx.problem_statement = first_user_text[:2000]

    # Aggregate files touched across all tool_use events.
    for ev in events:
        if ev.kind == "tool_use":
            ctx.files_touched.update(ev.files_touched)
    ctx.language = infer_language(ctx.repo, ctx.files_touched)

    return events, ctx


def events_to_text(events: Iterable[TranscriptEvent], *, tool_result_cap: int = 1200) -> str:
    """Render events back to a compact text block for LLM extraction.

    Truncates verbose tool_result bodies so prompts stay bounded. Keeps tool_use
    names + key inputs (file_path/command) because they are the strongest
    localization signal.
    """
    lines: list[str] = []
    for ev in events:
        if ev.kind == "text":
            if ev.text.strip():
                lines.append(f"[{ev.role}] {ev.text.strip()}")
        elif ev.kind == "thinking":
            # Keep thinking but mark it; it often contains the root-cause chain.
            t = ev.text.strip()
            if t:
                lines.append(f"[{ev.role}:thinking] {t[:600]}")
        elif ev.kind == "tool_use":
            inp = ev.tool_input
            key_parts = []
            for k in ("file_path", "path", "command", "pattern", "query"):
                if k in inp and inp[k]:
                    key_parts.append(f"{k}={str(inp[k])[:200]}")
            label = ", ".join(key_parts) if key_parts else ""
            lines.append(f"[{ev.role}:tool:{ev.tool_name}] {label}".rstrip())
        elif ev.kind == "tool_result":
            body = (ev.tool_result or "").strip()
            if body:
                lines.append(f"[tool_result] {body[:tool_result_cap]}")
    return "\n".join(lines)
