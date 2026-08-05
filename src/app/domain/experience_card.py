"""ExperienceCard: the structured unit of memory (CICL + MemGovern).

A card captures the *decision-relevant* essence of a coding session segment —
not the raw transcript. Five fields drive the decision-utility rerank (CICL
§3.1): trigger, root_cause, action_hint, files_symbols, failure_if_ignored.
Plus scope/temporal_anchor for filtering and gist/entities/keywords for record
retrieval.

The card serializes into `MemoryNote.type_specific` (backward-compatible with
the legacy profile/event/record shapes) so the persistence layer needs no
schema change for the card itself — only the version/supersedes columns are
added.

A zero-LLM heuristic extractor (`heuristic_card`) is provided so the pipeline
stays functional (and testable) without an API key. It is intentionally
conservative: it never fabricates root_cause/failure_if_ignored, only fills
what is recoverable from text + task context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .enums import MemoryType


@dataclass
class ExperienceCard:
    """Structured experience extracted from one transcript segment.

    All fields default to empty so a partial extraction is still valid. The
    `memory_type` drives which fields are emphasized downstream.
    """

    memory_type: MemoryType = MemoryType.RECORD
    # --- Provenance / retrieval marker ---
    # The SWEContextBench instance_id this experience came from. Embedded into
    # the composed content so retrieval can match on it (eval harness & cross-
    # task reuse). Empty for non-SWEContextBench adds.
    instance_id: str = ""
    # --- CICL 5 decision fields ---
    trigger: str = ""               # symptom / condition that fired the work
    root_cause: str = ""            # underlying cause
    action_hint: str = ""           # reusable fix / action
    files_symbols: list[str] = field(default_factory=list)  # touched files/symbols
    failure_if_ignored: str = ""    # what breaks if you ignore this
    # --- Scope / time ---
    scope: str = ""                 # repo / language / scenario
    temporal_anchor: str = ""       # version / date / commit
    # --- Record retrieval helpers ---
    gist: str = ""
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    # --- Profile pairs (stable facts) ---
    pairs: list[dict[str, str]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_type_specific(self) -> dict[str, Any]:
        """Serialize into `MemoryNote.type_specific`.

        We keep the legacy keys (pairs/trigger-action-outcome/gist-entities-keywords)
        for backward compatibility AND add the CICL fields under a `card` sub-dict
        so old readers still work while new readers see the full card.
        """
        ts: dict[str, Any] = {
            # Legacy-compatible keys
            "gist": self.gist,
            "entities": self.entities,
            "keywords": self.keywords,
            # CICL card
            "card": {
                "instance_id": self.instance_id,
                "trigger": self.trigger,
                "root_cause": self.root_cause,
                "action_hint": self.action_hint,
                "files_symbols": self.files_symbols,
                "failure_if_ignored": self.failure_if_ignored,
                "scope": self.scope,
                "temporal_anchor": self.temporal_anchor,
            },
        }
        if self.memory_type == MemoryType.PROFILE:
            ts["pairs"] = self.pairs
        if self.memory_type == MemoryType.EVENT:
            # Legacy event shape for old readers.
            ts["trigger"] = self.trigger
            ts["action"] = self.action_hint
            ts["outcome"] = self.root_cause or self.failure_if_ignored
            ts["temporal_anchor"] = self.temporal_anchor or "unspecified"
        return ts

    @classmethod
    def from_type_specific(
        cls, memory_type: MemoryType, ts: dict[str, Any] | None
    ) -> "ExperienceCard":
        """Inverse of `to_type_specific`. Tolerates legacy-only dicts."""
        ts = ts or {}
        card = ts.get("card") or {}
        # Recover legacy fields when the card sub-dict is absent.
        instance_id = card.get("instance_id", "")
        trigger = card.get("trigger") or ts.get("trigger", "")
        action_hint = card.get("action_hint") or ts.get("action", "")
        root_cause = card.get("root_cause") or ts.get("outcome", "")
        files_symbols = list(card.get("files_symbols") or [])
        failure_if_ignored = card.get("failure_if_ignored", "")
        scope = card.get("scope", "")
        temporal_anchor = card.get("temporal_anchor") or ts.get("temporal_anchor", "")
        gist = ts.get("gist", "")
        entities = list(ts.get("entities") or [])
        keywords = list(ts.get("keywords") or [])
        pairs = list(ts.get("pairs") or [])
        return cls(
            memory_type=memory_type,
            instance_id=instance_id,
            trigger=trigger,
            root_cause=root_cause,
            action_hint=action_hint,
            files_symbols=files_symbols,
            failure_if_ignored=failure_if_ignored,
            scope=scope,
            temporal_anchor=temporal_anchor,
            gist=gist,
            entities=entities,
            keywords=keywords,
            pairs=pairs,
        )

    def to_llm_dict(self) -> dict[str, Any]:
        """The JSON shape the LLM extractor is asked to return."""
        return {
            "memory_type": self.memory_type.value,
            "instance_id": self.instance_id,
            "trigger": self.trigger,
            "root_cause": self.root_cause,
            "action_hint": self.action_hint,
            "files_symbols": self.files_symbols,
            "failure_if_ignored": self.failure_if_ignored,
            "scope": self.scope,
            "temporal_anchor": self.temporal_anchor,
            "gist": self.gist,
            "entities": self.entities,
            "keywords": self.keywords,
            "pairs": self.pairs,
        }


# ---------------------------------------------------------------------------
# Zero-LLM heuristic
# ---------------------------------------------------------------------------

_EVENT_VERBS = re.compile(
    r"\b(fixed|fixes|resolved|patched|migrated|refactored|deployed|upgraded|"
    r"added|removed|replaced|reverted|disabled|enabled)\b",
    re.IGNORECASE,
)
_PROFILE_HINTS = re.compile(
    r"\b(uses?|built with|written in|framework|library|version|database|"
    r"deployed on|auth(?:entication)? with)\b",
    re.IGNORECASE,
)
_ERROR_HINTS = re.compile(
    r"\b(error|exception|traceback|npe|nullpointer|failed|failure|bug|crash|"
    r"attributeerror|typeerror|valueerror|keyerror|importerror)\b",
    re.IGNORECASE,
)
# File / symbol tokens: paths like a/b.py or CamelCase identifiers.
_FILE_TOKEN_RE = re.compile(r"[\w./\-]+/[./\w\-]*\.(?:py|js|ts|java|go|c|cpp|rb|php|rs)\b")
_SYMBOL_RE = re.compile(r"\b([A-Z][a-zA-Z0-9_]{2,}(?:\.[A-Z][a-zA-Z0-9_]+)*)\b")
_VERSION_RE = re.compile(r"\bv?(\d+\.\d+(?:\.\d+)?)\b")


def classify_heuristic(text: str) -> MemoryType:
    """Best-effort type classification without an LLM (LeanMem §3.2 style)."""
    if not text:
        return MemoryType.RECORD
    if _EVENT_VERBS.search(text):
        return MemoryType.EVENT
    if _PROFILE_HINTS.search(text) and not _ERROR_HINTS.search(text):
        return MemoryType.PROFILE
    return MemoryType.RECORD


def _extract_files_symbols(text: str, known_files: set[str]) -> list[str]:
    """Pull file paths and CamelCase symbols out of text."""
    out: list[str] = list(known_files)
    for m in _FILE_TOKEN_RE.finditer(text):
        out.append(m.group(0))
    seen: set[str] = set()
    uniq: list[str] = []
    for f in out:
        f = f.strip()
        if f and f not in seen:
            seen.add(f)
            uniq.append(f)
    # Cap to keep cards bounded.
    return uniq[:20]


def _first_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    for sep in (". ", ".\n", "!\n", "?\n", "\n"):
        idx = text.find(sep)
        if idx != -1:
            return text[: idx + 1].strip()
    return text[:300]


def heuristic_card(
    text: str,
    *,
    repo: str = "",
    language: str = "",
    files_touched: set[str] | None = None,
    temporal_anchor: str = "",
    instance_id: str = "",
) -> ExperienceCard:
    """Build a conservative ExperienceCard without an LLM call.

    Used as the fallback when `use_llm_on_add` is False or the LLM fails. Only
    fills fields recoverable from text + task context; leaves root_cause /
    failure_if_ignored empty when not evident (never fabricates).
    """
    files_touched = files_touched or set()
    memory_type = classify_heuristic(text)
    files_symbols = _extract_files_symbols(text, files_touched)
    scope_parts = [p for p in (repo, language) if p]
    scope = "/".join(scope_parts) if scope_parts else ""

    card = ExperienceCard(
        memory_type=memory_type,
        instance_id=instance_id,
        files_symbols=files_symbols,
        scope=scope,
        temporal_anchor=temporal_anchor,
        gist=_first_sentence(text) or text[:200],
    )

    if memory_type == MemoryType.EVENT:
        # trigger ≈ first error-ish phrase; action_hint ≈ the verb phrase.
        if _ERROR_HINTS.search(text):
            card.trigger = _first_sentence(text)
        card.action_hint = _first_sentence(text)
    elif memory_type == MemoryType.PROFILE:
        # Stable facts: gist carries the declarative sentence.
        card.pairs = []  # filled by profile_extractor in the service layer
    else:  # RECORD
        card.entities = _extract_symbols(text)
        card.keywords = _extract_keywords(text, files_symbols)

    return card


def _extract_symbols(text: str) -> list[str]:
    syms: list[str] = []
    seen: set[str] = set()
    for m in _SYMBOL_RE.finditer(text):
        s = m.group(1)
        if s not in seen and len(s) <= 40:
            seen.add(s)
            syms.append(s)
    return syms[:15]


_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "have", "was",
    "are", "but", "not", "you", "your", "use", "using", "can", "all",
    "The", "And", "For", "With", "That", "This", "But", "Was", "Are",
}


def _extract_keywords(text: str, files_symbols: list[str]) -> list[str]:
    """Cheap keyword extraction: capitalized/identifier tokens + file basenames."""
    kws: list[str] = list(files_symbols)
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\b", text):
        w = m.group(1)
        if w in _STOP or w.lower() in _STOP:
            continue
        kws.append(w)
    seen: set[str] = set()
    uniq: list[str] = []
    for k in kws:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq[:10]
