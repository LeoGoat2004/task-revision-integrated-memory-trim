"""Add pipeline: transcript → ExperienceCard → revision → persist.

Replaces the legacy append-and-summarize flow with an ingestion pipeline that
treats the /add payload as a real coding-agent transcript:

  1. `transcript_parser.parse_transcript` normalizes heterogeneous inputs
     (raw Claude Code JSONL rows OR normalized {role, content}) into a unified
     `TranscriptEvent` stream + `TaskContext` (instance_id, repo, files, lang).
  2. `chunker.segment_events` splits by task boundaries (TodoWrite / idle gap /
     size caps) instead of fixed message counts.
  3. Each segment → `ExperienceCard` via ONE LLM call
     (`EXTRACT_EXPERIENCE_CARD_PROMPT`). Falls back to the zero-LLM
     `heuristic_card` when the LLM is off / breaker tripped / call fails.
  4. `revision.decide_revision` reconciles the candidate against existing
     neighbours: exact-dup SKIP, near-neighbour judged supports/refines MERGE,
     contradicts SUPERSEDE, else INSERT. Keeps the memory base non-redundant.
  5. Embed the composed content; persist (insert / update / mark_superseded).
  6. For RECORD segments with file content in tool_results, extract tree-sitter
     symbols and upsert into `file_symbols` (feeds the Search symbol graph).

The pipeline stays online without an API key (heuristic cards + BM25-only
revision that never supersedes). Per-request LLM circuit breaker is reset at
entry and trips after `llm_failure_threshold` consecutive failures.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from ..config import get_domain_params, get_settings
from ..domain import prompts
from ..domain.chunker import segment_events
from ..domain.enums import MemoryType
from ..domain.experience_card import (
    ExperienceCard,
    classify_heuristic,
    heuristic_card,
)
from ..domain.models import MemoryNote
from ..domain.params import DomainParams
from ..domain.profile_extractor import extract_profile_pairs, merge_pairs, pairs_to_dict
from ..domain.revision import (
    RevisionAction,
    RevisionDecision,
    decide_revision,
    merge_cards,
)
from ..domain.transcript_parser import (
    TaskContext,
    TranscriptEvent,
    events_to_text,
    parse_transcript,
)
from ..infrastructure import embed as embed_client
from ..infrastructure import llm as llm_client
from ..infrastructure import sqlite as db
from ..infrastructure import tree_sitter as ts_adapter
from ..api.schemas import AddMessage


logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# ExperienceCard extraction (LLM + heuristic fallback)
# ---------------------------------------------------------------------------

def _llm_extract_card(
    segment_text: str, ctx: TaskContext, params: DomainParams
) -> ExperienceCard | None:
    """One LLM call → ExperienceCard. Returns None on any failure."""
    if not segment_text.strip():
        return None
    user_prompt = (
        f"Task context: repo={ctx.repo or 'unknown'} "
        f"language={ctx.language or 'unknown'} "
        f"files_touched={sorted(ctx.files_touched)[:20]}\n\n"
        f"Transcript segment:\n{segment_text[:params.transcript_max_segment_chars]}"
    )
    result = llm_client.chat_json(
        system=prompts.EXTRACT_EXPERIENCE_CARD_PROMPT,
        user=user_prompt,
        temperature=0.0,
        # Uses settings.llm_max_tokens (default 2048) — must be large enough
        # for the full ExperienceCard JSON. Don't hardcode here.
    )
    if not result or not isinstance(result, dict):
        return None
    try:
        memory_type = MemoryType(result.get("memory_type", "record"))
    except ValueError:
        memory_type = MemoryType.RECORD
    card = ExperienceCard(
        memory_type=memory_type,
        instance_id=ctx.instance_id,
        trigger=str(result.get("trigger", "") or "").strip(),
        root_cause=str(result.get("root_cause", "") or "").strip(),
        action_hint=str(result.get("action_hint", "") or "").strip(),
        files_symbols=[str(x).strip() for x in (result.get("files_symbols") or []) if isinstance(x, str) and str(x).strip()],
        failure_if_ignored=str(result.get("failure_if_ignored", "") or "").strip(),
        scope=str(result.get("scope", "") or "").strip(),
        temporal_anchor=str(result.get("temporal_anchor", "") or "").strip(),
        gist=str(result.get("gist", "") or "").strip(),
        entities=[str(x).strip() for x in (result.get("entities") or []) if isinstance(x, str) and str(x).strip()],
        keywords=[str(x).strip() for x in (result.get("keywords") or []) if isinstance(x, str) and str(x).strip()],
        pairs=[p for p in (result.get("pairs") or []) if isinstance(p, dict)],
    )
    # Backfill scope from task context if the LLM omitted it.
    if not card.scope:
        scope_parts = [p for p in (ctx.repo, ctx.language) if p]
        card.scope = "/".join(scope_parts) if scope_parts else ""
    if not card.temporal_anchor and ctx.base_commit:
        card.temporal_anchor = ctx.base_commit
    return card


def _build_card(
    segment_text: str, ctx: TaskContext, params: DomainParams, *, use_llm: bool
) -> ExperienceCard:
    """Extract a card, preferring the LLM and degrading to the heuristic."""
    if use_llm and llm_client.client_available() and not llm_client.breaker_tripped():
        card = _llm_extract_card(segment_text, ctx, params)
        if card is not None:
            return card
        logger.info("LLM card extraction failed; falling back to heuristic")
    return heuristic_card(
        segment_text,
        repo=ctx.repo,
        language=ctx.language,
        files_touched=set(ctx.files_touched),
        temporal_anchor=ctx.base_commit,
        instance_id=ctx.instance_id,
    )


# ---------------------------------------------------------------------------
# Content composition (the retrievable text stored in MemoryNote.content)
# ---------------------------------------------------------------------------

def _compose_content(card: ExperienceCard) -> str:
    """Compose a compact, retrievable text from a card's fields.

    This is what BM25 indexes and the dense embedder encodes — it must carry
    the discriminative lexical signal (file paths, identifiers, error names)
    plus the actionable gist. The full structured card is kept separately in
    `type_specific.card`.

    `instance_id` is deliberately embedded as a leading marker so that
    cross-task reuse and the eval harness can match a retrieved memory back to
    its source SWEContextBench experience by substring.
    """
    parts: list[str] = []
    if card.instance_id:
        parts.append(f"[{card.instance_id}]")
    if card.gist:
        parts.append(card.gist)
    if card.action_hint and card.action_hint != card.gist:
        parts.append(f"Action: {card.action_hint}")
    if card.root_cause:
        parts.append(f"Root cause: {card.root_cause}")
    if card.failure_if_ignored:
        parts.append(f"Risk: {card.failure_if_ignored}")
    if card.files_symbols:
        parts.append(f"Files: {', '.join(card.files_symbols[:12])}")
    if card.entities:
        parts.append(f"Entities: {', '.join(card.entities[:12])}")
    if card.keywords:
        parts.append(f"Keywords: {', '.join(card.keywords[:8])}")
    return " | ".join(parts) if parts else card.gist


# ---------------------------------------------------------------------------
# Relation judge (LLM-backed; injected into the pure revision module)
# ---------------------------------------------------------------------------

def _make_relation_judge() -> Callable[[str, str], tuple[str, dict]]:
    """Build an LLM-backed relation judge for revision.decide_revision."""
    def judge(candidate_text: str, existing_text: str) -> tuple[str, dict]:
        if not llm_client.client_available() or llm_client.breaker_tripped():
            return "unrelated", {}
        user_prompt = (
            f"NEW candidate experience:\n{candidate_text[:3000]}\n\n"
            f"EXISTING stored experience:\n{existing_text[:3000]}"
        )
        result = llm_client.chat_json(
            system=prompts.RELATION_JUDGE_PROMPT,
            user=user_prompt,
            temperature=0.0,
        )
        if not result or not isinstance(result, dict):
            return "unrelated", {}
        relation = str(result.get("relation", "unrelated")).strip().lower()
        merged = result.get("merged_fields") or {}
        if not isinstance(merged, dict):
            merged = {}
        return relation, merged
    return judge


# ---------------------------------------------------------------------------
# Symbol extraction from tool_result snippets
# ---------------------------------------------------------------------------

def _collect_file_contents(segment: list[TranscriptEvent]) -> dict[str, str]:
    """Pair each Read/Edit tool_use with its following tool_result content.

    Tree-sitter needs source text; the transcript only embeds it inside
    tool_result blocks. We pair a tool_use (which carries file_path) with the
    next tool_result in the segment to recover (file → content) for symbol
    extraction. Best-effort: missing pairs are silently skipped.
    """
    out: dict[str, str] = {}
    pending_file: str | None = None
    for ev in segment:
        if ev.kind == "tool_use" and ev.tool_name in ("Read", "Edit", "Write", "View"):
            fp = ev.tool_input.get("file_path") or ev.tool_input.get("path")
            if isinstance(fp, str) and fp.strip():
                pending_file = fp.strip()
        elif ev.kind == "tool_result" and pending_file:
            body = (ev.tool_result or "").strip()
            # Only keep non-trivial bodies that look like source (heuristic).
            if body and len(body) > 40:
                out[pending_file] = body
            pending_file = None
    return out


def _store_symbols(user_id: str, file_contents: dict[str, str]) -> None:
    """Extract + upsert symbols for each file with recovered content."""
    if not file_contents:
        return
    for file, content in file_contents.items():
        try:
            refs = ts_adapter.extract_symbols(file, content)
        except Exception as exc:  # noqa: BLE001 — never let symbol IO abort Add
            logger.debug("symbol extraction failed for %s: %s", file, exc)
            continue
        if not refs:
            continue
        lang = ts_adapter.lang_for_file(file) or ""
        db.upsert_file_symbols(
            user_id, file,
            [{"name": r.name, "kind": r.kind, "line": r.line, "node_type": r.node_type} for r in refs],
            lang,
        )


# ---------------------------------------------------------------------------
# Persistence helpers (insert / merge / supersede)
# ---------------------------------------------------------------------------

def _new_note(
    user_id: str, session_id: str, request_id: str,
    content: str, embedding: list[float], card: ExperienceCard,
) -> MemoryNote:
    return MemoryNote(
        id=f"mem_{uuid.uuid4().hex[:16]}",
        user_id=user_id,
        session_id=session_id,
        request_id=request_id,
        content=content,
        embedding=embedding,
        memory_type=card.memory_type,
        type_specific=card.to_type_specific(),
        created_at=_utcnow_iso(),
    )


def _apply_merge(
    target: MemoryNote, merged_card: ExperienceCard, content: str,
    embedding: list[float],
) -> None:
    """Persist a MERGE: bump version, replace content/card/embedding."""
    target.version += 1
    target.content = content
    target.embedding = embedding
    target.memory_type = merged_card.memory_type
    target.type_specific = merged_card.to_type_specific()
    target.created_at = _utcnow_iso()
    db.update_note(target)


def _apply_supersede(
    user_id: str, session_id: str, request_id: str,
    old: MemoryNote, content: str, embedding: list[float], card: ExperienceCard,
) -> MemoryNote:
    """Persist a SUPERSEDE: insert the new version, mark the old one superseded."""
    new_note = _new_note(user_id, session_id, request_id, content, embedding, card)
    new_note.version = old.version + 1
    new_note.supersedes_id = old.id
    db.insert_memory(new_note)
    db.mark_superseded(old.id, new_note.id)
    return new_note


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def add(
    user_id: str,
    session_id: str,
    request_id: str,
    messages: list[AddMessage],
) -> int:
    """Run the full Add pipeline. Returns the number of memories written.

    Fail-closed semantics: LLM/embed failures degrade to the zero-LLM heuristic
    path rather than aborting, so the contract "Add processes successfully and
    is retrievable" holds. Exact-duplicate segments are skipped (ingestion, not
    append); merges and supersedes count as one write each.
    """
    raw_messages = [m.model_dump() for m in messages]
    return add_raw(user_id, session_id, request_id, raw_messages)


def add_raw(
    user_id: str,
    session_id: str,
    request_id: str,
    raw_messages: list[dict],
) -> int:
    """Ingest a raw transcript (list of dict rows) without the wire schema.

    Accepts both raw Claude Code JSONL rows (dicts with `type`/`message`) and
    normalized `{role, content}` dicts — `parse_transcript` tolerates either.
    Used by the eval harness and any batch loader that has the raw transcript
    and wants to preserve tool_use/tool_result structure (the `{role, content:
    str}` wire contract would otherwise flatten tool blocks to text).
    """
    settings = get_settings()
    params = get_domain_params()
    llm_client.reset_breaker()
    use_llm = bool(settings.use_llm_on_add)

    # 1. Parse the heterogeneous input into a unified event stream + task ctx.
    events, ctx = parse_transcript(raw_messages)
    if not events:
        return 0

    # 2. Segment by task boundary.
    segments = segment_events(events, params)
    if not segments:
        return 0

    existing = db.fetch_recent_memories(user_id, limit=params.dedup_neighbor_cap)
    relation_judge = _make_relation_judge() if use_llm else None
    written = 0

    for segment in segments:
        segment_text = events_to_text(
            segment, tool_result_cap=params.transcript_tool_result_cap
        )
        if not segment_text.strip():
            continue

        # 3. Extract an ExperienceCard (LLM or heuristic).
        card = _build_card(segment_text, ctx, params, use_llm=use_llm)

        # PROFILE-type: fold regex-extracted stable pairs into the card so the
        # cumulative user profile accumulates across adds (LeanMem §3.2).
        if card.memory_type == MemoryType.PROFILE:
            new_pairs = pairs_to_dict(extract_profile_pairs(segment_text))
            existing_profiles = db.fetch_memories_by_type(user_id, MemoryType.PROFILE.value)
            cumulative: list[dict[str, str]] = []
            for prof in existing_profiles:
                cumulative = merge_pairs(cumulative, prof.type_specific.get("pairs", []))
            card.pairs = merge_pairs(cumulative, new_pairs)

        content = _compose_content(card)
        if not content.strip():
            content = segment_text[:500]

        # 4. Embed the composed content (zero-vector when no API key).
        embeddings = embed_client.embed_texts([content])
        embedding = embeddings[0] if embeddings else []

        # 5. Revision: dedup / merge / supersede / insert.
        decision: RevisionDecision = decide_revision(
            candidate_content=content,
            candidate_embedding=embedding,
            candidate_card=card,
            existing=existing,
            params=params,
            relation_judge=relation_judge,
            user_id=user_id,
        )

        if decision.action == RevisionAction.SKIP:
            logger.info("Skip duplicate memory for user_id=%s: %s", user_id, decision.reason)
            continue

        if decision.action == RevisionAction.MERGE and decision.target_id:
            target = next((n for n in existing if n.id == decision.target_id), None)
            if target is None:
                # Target vanished (concurrent delete) — fall back to insert.
                decision = RevisionDecision(action=RevisionAction.INSERT, reason="merge target missing")
            else:
                merged_card = decision.merged_card or card
                merged_content = _compose_content(merged_card)
                merged_embedding = embedding
                _apply_merge(target, merged_card, merged_content, merged_embedding)
                existing = [n for n in existing if n.id != target.id]
                existing.insert(0, target)
                written += 1
                logger.info("Merged memory into %s for user_id=%s", target.id, user_id)
                continue

        if decision.action == RevisionAction.SUPERSEDE and decision.target_id:
            old = next((n for n in existing if n.id == decision.target_id), None)
            if old is None:
                decision = RevisionDecision(action=RevisionAction.INSERT, reason="supersede target missing")
            else:
                new_note = _apply_supersede(
                    user_id, session_id, request_id, old, content, embedding, card
                )
                existing.insert(0, new_note)
                written += 1
                logger.info("Superseded memory %s → %s for user_id=%s", old.id, new_note.id, user_id)
                continue

        # INSERT (default or fallback from missing merge/supersede target).
        note = _new_note(user_id, session_id, request_id, content, embedding, card)
        db.insert_memory(note)
        existing.insert(0, note)
        written += 1

        # 6. Symbol graph: extract from recovered file contents (RECORD-type
        #    segments carry the richest source signal).
        if card.memory_type == MemoryType.RECORD and card.files_symbols:
            file_contents = _collect_file_contents(segment)
            _store_symbols(user_id, file_contents)

    return written
