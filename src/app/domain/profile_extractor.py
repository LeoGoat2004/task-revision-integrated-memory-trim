"""Heuristic (rule-based) profile extractor.

The LLM-based extraction in `EXTRACT_PROFILE_PROMPT` is preferred when available.
This module is the zero-LLM fallback that LeanMem's paper relies on for the
"profile" type: stable, declarative project facts are often recoverable with
simple regex patterns, so burning an LLM call for them is wasteful.

Output is a list of `(attribute, value)` pairs deduplicated against any
previously-stored pairs in the same user_id scope.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Tuple


# Each pattern is `(attribute_name, compiled_regex)`. The regex must capture
# the value in group 1. Order matters: earlier patterns win on overlap.
_PROFILE_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    (
        "language",
        re.compile(
            r"\b(?:uses?|written in|built with|implemented in)\s+"
            r"(python|java|javascript|typescript|go|rust|c\+\+|c#|ruby|php|swift|kotlin|scala)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "framework",
        re.compile(
            r"\b(?:uses?|using|built with|on top of)\s+"
            r"([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)?)\s+(?:framework|library|sdk)",
        ),
    ),
    (
        "database",
        re.compile(
            r"\b(?:uses?|with|stored in|backend)\s+"
            r"(postgres(?:ql)?|mysql|mariadb|mongodb|redis|sqlite|elasticsearch|"
            r"dynamodb|cassandra|influxdb|clickhouse)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "auth",
        re.compile(
            r"\bauth(?:entication)?\s+(?:uses?|with)\s+"
            r"(jwt|oauth2?|oauth|session|basic|saml|api[_ ]?key|cookie)",
            re.IGNORECASE,
        ),
    ),
    (
        "version",
        re.compile(
            r"\b(?:version|v)\s*(\d+\.\d+(?:\.\d+)?)\b", re.IGNORECASE
        ),
    ),
    (
        "deployment",
        re.compile(
            r"\b(?:deploy(?:ed|ment)?\s+(?:on|to|via|using))\s+"
            r"(docker|kubernetes|k8s|aws|gcp|azure|vercel|fly\.io|heroku|railway)",
            re.IGNORECASE,
        ),
    ),
]


def extract_profile_pairs(content: str) -> List[Tuple[str, str]]:
    """Extract up to 5 (attribute, value) pairs from a profile-eligible text.

    Returns a list of `(attr, value)` tuples. The original token piece is
    stored as the value (lowercased) for stable equality checks.
    """
    if not content:
        return []

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for attr, pattern in _PROFILE_PATTERNS:
        for match in pattern.finditer(content):
            value = match.group(1).strip().lower()
            if not value:
                continue
            pair = (attr, value)
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
            if len(pairs) >= 5:
                return pairs
    return pairs


def pairs_to_dict(pairs: Iterable[Tuple[str, str]]) -> list[dict[str, str]]:
    """Convert an iterable of (attr, value) tuples to the JSON-serializable
    list-of-dicts format expected by `MemoryNote.type_specific` for profile."""
    return [{"attr": attr, "value": value} for attr, value in pairs]


def merge_pairs(
    existing: list[dict[str, str]],
    new: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Merge two profile pair lists, deduplicating by (attr, value)."""
    seen: set[tuple[str, str]] = {(p["attr"], p["value"]) for p in existing}
    merged = list(existing)
    for p in new:
        key = (p["attr"], p["value"])
        if key in seen:
            continue
        seen.add(key)
        merged.append(p)
    return merged
