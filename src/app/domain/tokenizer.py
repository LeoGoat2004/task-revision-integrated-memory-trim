"""Code-aware tokenizer for BM25 retrieval.

Design rationale (codifies "identifier-first" retrieval from Aider's repomap
and the LeanMem-era code-memory literature):

1. Preserve source identifiers as-is (camelCase, snake_case, dot.path, hyphen).
2. Split compound identifiers into sub-tokens so partial matches work.
3. Keep both the original token AND its sub-tokens to maximize recall.
4. Filter trivial tokens (length < 2, pure digits).
"""
from __future__ import annotations

import re

# A run of ASCII identifier characters OR a run of CJK characters.
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*+|[\u4e00-\u9fff]+")

# Boundary: lower-case / digit followed by upper-case (camelCase).
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# Boundary: acronym + capitalized word (e.g., HTTPRequest -> HTTP, Request).
_ACRONYM_BOUNDARY_RE = re.compile(r"([A-Z]+)([A-Z][a-z])")

# Identifiers may be joined by these characters.
_JOINER_RE = re.compile(r"[._\-]+")


def _split_identifier(token: str) -> list[str]:
    """Split a single identifier into sub-tokens while preserving the original.

    Returns a list of unique sub-tokens (set semantics). The original token
    itself is always included.
    """
    parts = _JOINER_RE.split(token)
    pieces: set[str] = set()
    for part in parts:
        if not part:
            continue
        # Keep the original token piece (lowercased).
        pieces.add(part.lower())
        # Acronym boundary: HTTPRequest -> HTTP, Request.
        expanded = _ACRONYM_BOUNDARY_RE.sub(r"\1 \2", part)
        # camelCase boundary: getUserId -> get User Id.
        expanded = _CAMEL_BOUNDARY_RE.sub(" ", expanded)
        for piece in expanded.split():
            pieces.add(piece.lower())
    return list(pieces) if pieces else [token.lower()]


def tokenize(text: str) -> list[str]:
    """Tokenize a text for BM25 against code-engineering content.

    Rules:
      - Runs of ASCII identifiers OR runs of CJK characters become tokens.
      - Compound identifiers (camelCase, snake_case, dot.path, hyphen) are
        split and all sub-tokens are kept (maximizes partial-match recall).
      - Tokens of length < 2 are dropped.
      - Pure-digit tokens are dropped.
    """
    if not text:
        return []
    out: list[str] = []
    for match in _TOKEN_RE.findall(text):
        if match.isdigit():
            continue
        if len(match) < 2:
            continue
        out.extend(_split_identifier(match))
    return out
