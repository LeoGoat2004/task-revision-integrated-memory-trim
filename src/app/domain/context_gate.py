"""Context gate: intent-aware filtering + multi-granularity packaging.

Inspired by ContextSniper (L0/L1/L2 views) and Anthropic's just-in-time
retrieval (return an identifier first, load full text on demand). The gate
sits at the *output* of the Search pipeline and:

  1. Filters candidates whose memory type is irrelevant to the query intent
     (e.g. a profile-type memory returned for an event-intent query is dropped
     unless it also scores high on utility).
  2. Packages each surviving result at the requested granularity:
       L0 — one-liner: "{trigger} → {action_hint}" (bounded to l0_max_chars).
       L1 — structured 5 fields (the default; compact yet actionable).
       L2 — full content (record-type detail; for the repair phase).

Pure domain: operates on SearchResult + the card recovered from type_specific.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .enums import MemoryType
from .experience_card import ExperienceCard
from .models import SearchResult
from .params import DomainParams, DEFAULT_PARAMS


VALID_GRANULARITIES = ("L0", "L1", "L2")


@dataclass
class GatedResult:
    """A search result after context-gate packaging."""

    id: str
    content: str              # the granularity-specific payload
    memory_type: MemoryType
    score: float
    created_at: str
    granularity: str          # L0 | L1 | L2
    card: ExperienceCard | None = None  # full card for callers that want it


def _coerce_granularity(g: str | None, params: DomainParams) -> str:
    if g and g.upper() in VALID_GRANULARITIES:
        return g.upper()
    return params.default_granularity.upper()


def _l0_text(card: ExperienceCard, note_content: str, params: DomainParams) -> str:
    parts = []
    if card.instance_id:
        parts.append(f"[{card.instance_id}]")
    if card.trigger:
        parts.append(card.trigger)
    if card.action_hint:
        parts.append("→ " + card.action_hint)
    text = " ".join(parts) if parts else (card.gist or note_content)
    if len(text) > params.l0_max_chars:
        text = text[: params.l0_max_chars - 1].rstrip() + "…"
    return text


def _l1_text(card: ExperienceCard) -> str:
    """Structured 5-field view, one line each (compact yet actionable)."""
    lines = []
    if card.instance_id:
        lines.append(f"instance: {card.instance_id}")
    if card.trigger:
        lines.append(f"trigger: {card.trigger}")
    if card.root_cause:
        lines.append(f"root_cause: {card.root_cause}")
    if card.action_hint:
        lines.append(f"action: {card.action_hint}")
    if card.files_symbols:
        lines.append(f"files: {', '.join(card.files_symbols[:8])}")
    if card.failure_if_ignored:
        lines.append(f"risk: {card.failure_if_ignored}")
    if not lines:
        # Fall back to gist / entities for record-type memories.
        if card.gist:
            lines.append(card.gist)
        if card.entities:
            lines.append(f"entities: {', '.join(card.entities[:8])}")
    return "\n".join(lines)


def _l2_text(card: ExperienceCard, note_content: str) -> str:
    """Full content: the stored memory text plus the structured fields header."""
    header = _l1_text(card)
    body = note_content.strip()
    if header and body:
        return f"{header}\n---\n{body}"
    return body or header


def package(
    result: SearchResult,
    card: ExperienceCard | None,
    note_content: str,
    granularity: str | None,
    params: DomainParams = DEFAULT_PARAMS,
) -> GatedResult:
    """Package a single result at the requested granularity."""
    g = _coerce_granularity(granularity, params)
    if card is None:
        # No card (legacy note) — fall back to the raw content at any level.
        return GatedResult(
            id=result.id, content=note_content, memory_type=result.memory_type,
            score=result.score, created_at=result.created_at, granularity=g,
        )
    if g == "L0":
        text = _l0_text(card, note_content, params)
    elif g == "L2":
        text = _l2_text(card, note_content)
    else:  # L1
        text = _l1_text(card) or note_content
    return GatedResult(
        id=result.id, content=text, memory_type=result.memory_type,
        score=result.score, created_at=result.created_at, granularity=g,
        card=card,
    )


def gate(
    query_intent_weights: dict[MemoryType, float] | None,
    results: Sequence[SearchResult],
    notes_by_id: dict[str, "MemoryNote"],  # noqa: F821 — forward ref
    *,
    granularity: str | None = None,
    params: DomainParams = DEFAULT_PARAMS,
    keep_min: int = 1,
) -> list[GatedResult]:
    """Filter + package the reranked results.

    The gate drops candidates whose type carries zero intent weight AND whose
    score is in the bottom tier (we never drop below `keep_min` results so the
    caller always gets something when memories exist).
    """
    # If no intent weights, keep everything (no filtering signal).
    if not query_intent_weights:
        keep = list(results)
    else:
        zero_weight_types = {t for t, w in query_intent_weights.items() if w <= 0.0}
        # Keep results whose type has positive weight OR which are in the top
        # half by score (avoid over-filtering when the classifier is wrong).
        if results:
            scores = [r.score for r in results]
            median = sorted(scores)[len(scores) // 2]
        else:
            median = 0.0
        keep = [
            r for r in results
            if r.memory_type not in zero_weight_types or r.score >= median
        ]
        if len(keep) < keep_min:
            keep = list(results)

    out: list[GatedResult] = []
    for r in keep:
        note = notes_by_id.get(r.id)
        card = None
        content = r.content
        if note is not None:
            card = ExperienceCard.from_type_specific(note.memory_type, note.type_specific)
            content = note.content
        out.append(package(r, card, content, granularity, params))
    return out
