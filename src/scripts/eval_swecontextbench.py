"""SWEContextBench evaluation harness for the code-memory system.

Evaluates retrieval quality (not end-to-end agent solving): does /search surface
the *relevant* past experience for a given base task?

Ground-truth relevance (SWEContextBench annotation spirit + Sourcegraph
localization stratification):
  - strong : base task and past-experience share the same repo AND overlap in
             source files touched (the experience is directly transferable).
  - weak   : same repo, no file overlap (transferable at the repo/convention
             level only).
  - none   : different repo.

Metrics:
  - recall@k        : fraction of base tasks whose top-k results contain ≥1
                      strong-relevant memory.
  - precision@k     : share of top-k results that are strong- or weak-relevant.
  - strong_hit@1    : top-1 result is strong-relevant.
  - mrr             : reciprocal rank of the first strong-relevant result.
  Stratified by localization difficulty (patch file count): easy / medium / hard.

Modes:
  - direct (default): calls add_service / search_service in-process. Fast, no
    server needed. Uses the zero-LLM heuristic path when no API key is set —
    this is the smoke validation required before real keys.
  - http: talks to a running server (SMOKE_URL) — use for the real-key run.

Usage:
    python src/scripts/eval_swecontextbench.py --limit 20
    python src/scripts/eval_swecontextbench.py --mode http --top-k 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean

# Make `app.*` importable when run as a script.
_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

import httpx  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.domain.transcript_parser import parse_transcript  # noqa: E402


BENCH = Path("G:/Projects/hackathon/AgentMemory2608/third_party/SWEContextBench/cases")
PE_DIR = BENCH / "SWEContextBench Lite Past Experience"
LITE_DIR = BENCH / "SWEContextBench Lite"

# Files in PE transcripts are prefixed with the swebench testbed path; strip it
# to recover the repo-relative source path. Exclude tests / artifacts.
_TESTBED_PREFIXES = (
    "./swebench_9_15/testbed/",
    "swebench_9_15/testbed/",
    "./swebench_9_15/",
)
_EXCLUDE_SUBSTR = ("/tests/", "/test_", "test_", "/tests", "manual.yaml",
                   "/output/", "swebench_", "conftest", ".pyc")

# Patch file extraction from a unified diff.
_DIFF_FILE_RE = re.compile(r"^diff --git a/(?P<f>.+?) b/(?P<g>.+)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class ExperienceRecord:
    file: str            # PE jsonl filename (hash)
    instance_id: str
    repo: str
    source_files: set[str] = field(default_factory=set)
    n_rows: int = 0


@dataclass
class BaseTask:
    instance_id: str
    repo: str
    problem_statement: str
    patch_files: set[str] = field(default_factory=set)


def _normalize_source_path(raw: str) -> str:
    """Strip swebench testbed prefixes; return '' if it's a test/artifact."""
    p = raw.replace("\\", "/").strip()
    for prefix in _TESTBED_PREFIXES:
        if p.startswith(prefix):
            p = p[len(prefix):]
            break
    # Drop leading "./"
    p = p.lstrip("./")
    low = p.lower()
    if any(x in low for x in _EXCLUDE_SUBSTR):
        return ""
    # Glob patterns (e.g. "*.py", "*/mixture/tests/*.py") aren't real files.
    if "*" in p:
        return ""
    return p


def load_experiences() -> list[ExperienceRecord]:
    out: list[ExperienceRecord] = []
    for p in sorted(PE_DIR.glob("*.jsonl")):
        rows: list[dict] = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not rows:
            continue
        _events, ctx = parse_transcript(rows)
        src = {_normalize_source_path(f) for f in ctx.files_touched}
        src.discard("")
        out.append(ExperienceRecord(
            file=p.name, instance_id=ctx.instance_id, repo=ctx.repo,
            source_files=src, n_rows=len(rows),
        ))
    return out


def load_base_tasks() -> list[BaseTask]:
    out: list[BaseTask] = []
    for p in sorted(LITE_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        patch = d.get("patch", "") or ""
        files = set(_DIFF_FILE_RE.findall(patch))
        # findall returns tuples (f, g); they should match — take the first.
        patch_files = {m[0] for m in _DIFF_FILE_RE.findall(patch)}
        out.append(BaseTask(
            instance_id=d.get("instance_id", p.stem),
            repo=d.get("repo", ""),
            problem_statement=(d.get("problem_statement") or "")[:1500],
            patch_files=patch_files,
        ))
    return out


# ---------------------------------------------------------------------------
# Ground-truth relevance
# ---------------------------------------------------------------------------

def relevance(task: BaseTask, exp: ExperienceRecord) -> str:
    """Return 'strong' | 'weak' | 'none'."""
    if not exp.repo or exp.repo != task.repo:
        return "none"
    if exp.source_files and task.patch_files and (exp.source_files & task.patch_files):
        return "strong"
    return "weak"


def localization_tier(task: BaseTask) -> str:
    """Sourcegraph-style localization difficulty from patch breadth."""
    n = len(task.patch_files)
    if n <= 2:
        return "easy"
    if n <= 5:
        return "medium"
    return "hard"


# ---------------------------------------------------------------------------
# Eval driver
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    instance_id: str
    tier: str
    n_strong: int
    n_weak: int
    n_none: int
    strong_hit_at_1: bool
    recall_at_k: bool
    precision_at_k: float
    first_strong_rr: float  # 1/rank of first strong hit, 0 if none


def _build_query(task: BaseTask) -> str:
    """Retrieval query: repo + problem statement (the agent's task framing)."""
    ps = task.problem_statement.strip().replace("\n", " ")
    repo = task.repo.split("/")[-1] if task.repo else ""
    return f"{repo} {ps[:600]}".strip()


def transcript_to_wire(rows: list[dict]) -> list[dict]:
    """Adapt raw Claude Code JSONL transcript rows to the project's wire format.

    The project's /add API contract is `{role: str, content: str}` — it does
    NOT accept raw transcript blocks. This adapter lives in the EVAL harness
    (not in the project API) so the API stays clean and standard; the bench is
    just a data source here.

    To preserve the file-path / tool-action signal that retrieval depends on,
    tool_use blocks are serialized to text like `[tool_use: Read(file_path=...)]`
    so the parser's regex and BM25 can still index them. tool_result blocks are
    included truncated. Non-conversation rows (file-history-snapshot, etc.) are
    skipped.
    """
    wire: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        # Resolve role: prefer row["type"], fall back to message.role, then "user".
        role = row.get("type") or ""
        msg = row.get("message") if isinstance(row.get("message"), dict) else None
        if msg and not role:
            role = msg.get("role", "")
        # Normalize role to user/assistant.
        if role not in ("user", "assistant"):
            continue  # skip file-history-snapshot, summary, etc.
        # Resolve content: row["content"] (normalized) or msg["content"] (raw).
        content = row.get("content")
        if content is None and msg is not None:
            content = msg.get("content")
        text = _blocks_to_text(content)
        if text.strip():
            wire.append({"role": role, "content": text})
    return wire


# Tool-result text cap: keep enough for BM25 indexing without blowing up tokens.
_TOOL_RESULT_CAP = 600


def _blocks_to_text(content: object) -> str:
    """Serialize a message's content (str or block list) to a single text string."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for blk in content:
        if not isinstance(blk, dict):
            continue
        btype = blk.get("type", "")
        if btype == "text":
            parts.append(blk.get("text", ""))
        elif btype == "tool_use":
            name = blk.get("name", "")
            inp = blk.get("input") if isinstance(blk.get("input"), dict) else {}
            # Surface the high-signal args (file_path, path, pattern, command).
            key_args = []
            for k in ("file_path", "path", "pattern", "command", "query", "url"):
                if k in inp and inp[k]:
                    key_args.append(f"{k}={inp[k]}")
            sig = ", ".join(key_args)
            parts.append(f"[tool_use: {name}({sig})]" if sig else f"[tool_use: {name}]")
        elif btype == "tool_result":
            rc = blk.get("content")
            if isinstance(rc, str):
                rtext = rc
            elif isinstance(rc, list):
                rtext = " ".join(
                    b.get("text", "") for b in rc
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                rtext = ""
            if rtext:
                parts.append(f"[tool_result] {rtext[:_TOOL_RESULT_CAP]}")
        elif btype == "thinking":
            # Skip thinking blocks — internal reasoning, not retrievable signal.
            pass
    return "\n".join(p for p in parts if p)


class DirectBackend:
    """In-process backend: calls the service layer's public `add()` (no HTTP).

    Uses the SAME wire format ({role, content:str}) as the HTTP API — the raw
    transcript is adapted by `transcript_to_wire()` first. This means the eval
    exercises the identical code path as a real platform /add call, not a
    special internal shortcut.
    """

    def __init__(self, user_id: str) -> None:
        from app.infrastructure import sqlite as db
        from app.services import add_service, search_service
        self._db = db
        self._add = add_service
        self._search = search_service
        self.user_id = user_id
        db.init_db()

    def ingest(self, messages: list[dict], session_id: str) -> int:
        from app.api.schemas import AddMessage
        wire = transcript_to_wire(messages)
        if not wire:
            return 0
        msgs = [AddMessage(role=m["role"], content=m["content"]) for m in wire]
        return self._add.add(
            user_id=self.user_id, session_id=session_id,
            request_id=f"eval:{session_id}", messages=msgs,
        )

    def search(self, query: str, top_k: int) -> list[dict]:
        results = self._search.search(
            user_id=self.user_id, query=query, top_k=top_k,
        )
        out = []
        for r in results:
            # Surface the card's files_symbols as a separate match target so
            # relevance can hit even when the composed content abbreviates paths.
            files_syms: list[str] = []
            card = getattr(r, "card", None)
            if card is not None:
                files_syms = list(getattr(card, "files_symbols", []) or [])
            out.append({
                "id": r.id, "content": r.content,
                "memory_type": r.memory_type.value,
                "files_symbols": files_syms,
            })
        return out


class HttpBackend:
    """HTTP backend: talks to a running code-memory server via the real API.

    Uses the same `transcript_to_wire()` adapter as DirectBackend, so both
    backends exercise the identical wire contract.
    """

    def __init__(self, user_id: str, base_url: str, api_key: str) -> None:
        self.user_id = user_id
        self._client = httpx.Client(base_url=base_url, timeout=60.0)
        self._headers = {"X-Api-Key": api_key}

    def ingest(self, messages: list[dict], session_id: str) -> int:
        wire = transcript_to_wire(messages)
        if not wire:
            return 0
        r = self._client.post("/add", json={
            "request_id": f"eval:{session_id}",
            "messages": wire,
            "user_id": self.user_id,
            "session_id": session_id,
        }, headers=self._headers)
        r.raise_for_status()
        # /add doesn't return count; approximate by success.
        return 1 if r.json().get("success") else 0

    def search(self, query: str, top_k: int) -> list[dict]:
        r = self._client.post("/search", json={
            "query": query, "user_id": self.user_id,
            "top_k": top_k,
        }, headers=self._headers)
        r.raise_for_status()
        return r.json().get("data", [])


def run_eval(
    backend, experiences: list[ExperienceRecord],
    tasks: list[BaseTask], top_k: int, task_limit: int,
    pe_limit: int = 0,
) -> tuple[list[TaskResult], dict]:
    # Map memory content → experience for relevance lookup. We ingest each PE
    # transcript and remember its instance_id; search results are matched back
    # by checking whether the returned content mentions the experience's
    # instance_id or its source files (the heuristic card embeds these).
    pe_to_ingest = experiences[:pe_limit] if pe_limit > 0 else experiences
    print(f"[1/3] Ingesting {len(pe_to_ingest)} past-experience transcripts...")
    t0 = time.time()
    ingested = 0
    for i, exp in enumerate(pe_to_ingest, 1):
        rows: list[dict] = []
        with open(PE_DIR / exp.file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        n = backend.ingest(rows, session_id=f"pe:{exp.file}")
        if n:
            ingested += 1
        if i % 10 == 0 or i == len(pe_to_ingest):
            elapsed = time.time() - t0
            print(f"      [{i}/{len(pe_to_ingest)}] ingested={ingested} elapsed={elapsed:.0f}s", flush=True)
    print(f"      ingested {ingested}/{len(pe_to_ingest)} in {time.time()-t0:.1f}s")

    print(f"[2/3] Searching {len(tasks)} base tasks (limit={task_limit})...")
    results: list[TaskResult] = []
    eval_tasks = tasks[:task_limit] if task_limit > 0 else tasks
    for task in eval_tasks:
        query = _build_query(task)
        hits = backend.search(query, top_k)
        # Score relevance of each hit against ALL experiences, taking the max
        # relevance a hit could correspond to (a hit is "strong" if it matches
        # any strong-relevant experience for this task).
        hit_rels: list[str] = []
        for h in hits:
            # Match target = composed content + the card's files_symbols (raw
            # testbed paths). The normalized exp.source_files are substrings of
            # the raw paths, so `in` suffices for file overlap.
            content = (h.get("content") or "").lower()
            files_syms = h.get("files_symbols") or []
            target = content + " " + " ".join(f.lower() for f in files_syms)
            best = "none"
            for exp in pe_to_ingest:
                rel = relevance(task, exp)
                if rel == "none":
                    continue
                # A hit matches an experience if it mentions the experience's
                # instance_id or any of its source files.
                markers = [exp.instance_id.lower()] + [f.lower() for f in exp.source_files]
                if any(mk and mk in target for mk in markers):
                    if rel == "strong":
                        best = "strong"
                        break
                    if rel == "weak" and best != "strong":
                        best = "weak"
            hit_rels.append(best)

        n_strong = sum(1 for r in hit_rels if r == "strong")
        n_weak = sum(1 for r in hit_rels if r == "weak")
        n_none = sum(1 for r in hit_rels if r == "none")
        first_strong_rr = 0.0
        for i, r in enumerate(hit_rels, 1):
            if r == "strong":
                first_strong_rr = 1.0 / i
                break
        results.append(TaskResult(
            instance_id=task.instance_id,
            tier=localization_tier(task),
            n_strong=n_strong, n_weak=n_weak, n_none=n_none,
            strong_hit_at_1=bool(hit_rels) and hit_rels[0] == "strong",
            recall_at_k=n_strong > 0,
            precision_at_k=(n_strong + n_weak) / max(1, len(hit_rels)),
            first_strong_rr=first_strong_rr,
        ))

    # Aggregate.
    print("[3/3] Aggregating metrics...")
    n = max(1, len(results))
    summary = {
        "n_tasks": len(results),
        "top_k": top_k,
        "recall_at_k": sum(r.recall_at_k for r in results) / n,
        "precision_at_k": mean(r.precision_at_k for r in results) if results else 0.0,
        "strong_hit_at_1": sum(r.strong_hit_at_1 for r in results) / n,
        "mrr": mean(r.first_strong_rr for r in results) if results else 0.0,
        "by_tier": {},
    }
    for tier in ("easy", "medium", "hard"):
        tier_rs = [r for r in results if r.tier == tier]
        tn = max(1, len(tier_rs))
        summary["by_tier"][tier] = {
            "n": len(tier_rs),
            "recall_at_k": sum(r.recall_at_k for r in tier_rs) / tn,
            "strong_hit_at_1": sum(r.strong_hit_at_1 for r in tier_rs) / tn,
            "mrr": mean(r.first_strong_rr for r in tier_rs) if tier_rs else 0.0,
        }
    return results, summary


def main() -> int:
    ap = argparse.ArgumentParser(description="SWEContextBench retrieval harness")
    ap.add_argument("--mode", choices=["direct", "http"], default="direct")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=20,
                    help="base tasks to evaluate (0 = all)")
    ap.add_argument("--pe-limit", type=int, default=0,
                    help="past-experience transcripts to ingest (0 = all 300). "
                         "Use a small number (e.g. 30) for a fast LLM-on smoke.")
    ap.add_argument("--user-id", default="eval:swecontextbench")
    ap.add_argument("--report", default="eval_report.json")
    args = ap.parse_args()

    experiences = load_experiences()
    tasks = load_base_tasks()
    print(f"Loaded {len(experiences)} experiences, {len(tasks)} base tasks")

    if args.mode == "direct":
        backend = DirectBackend(args.user_id)
    else:
        s = get_settings()
        base = os.environ.get("SMOKE_URL", f"http://127.0.0.1:{s.port}")
        backend = HttpBackend(args.user_id, base, s.memory_system_key)

    results, summary = run_eval(backend, experiences, tasks, args.top_k, args.limit, args.pe_limit)

    report = {
        "config": {"mode": args.mode, "top_k": args.top_k, "limit": args.limit,
                   "pe_limit": args.pe_limit, "user_id": args.user_id},
        "summary": summary,
        "tasks": [asdict(r) for r in results],
    }
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(f"tasks={summary['n_tasks']} top_k={summary['top_k']}")
    print(f"recall@{args.top_k}      = {summary['recall_at_k']:.3f}")
    print(f"precision@{args.top_k}   = {summary['precision_at_k']:.3f}")
    print(f"strong_hit@1   = {summary['strong_hit_at_1']:.3f}")
    print(f"MRR            = {summary['mrr']:.3f}")
    for tier, m in summary["by_tier"].items():
        print(f"  [{tier}] n={m['n']} recall={m['recall_at_k']:.3f} "
              f"hit@1={m['strong_hit_at_1']:.3f} mrr={m['mrr']:.3f}")
    print(f"\nReport written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
