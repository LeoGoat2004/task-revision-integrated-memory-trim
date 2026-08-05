"""All LLM prompts for the code memory system.

Design rules:
- ALL prompts are in English (matches the underlying model's training distribution
  and is the standard for production code-memory systems).
- Every prompt is a module-level constant so it can be unit-tested and tuned
  without touching the call site.
- Output formats requested explicitly (JSON Schema hints in the prompt body).
"""
from __future__ import annotations

# ----------------------------------------------------------------------------
# Add pipeline
# ----------------------------------------------------------------------------

# Concatenate messages and produce a single structured "experience entry".
# Used as the source content for every memory, regardless of type.
SUMMARIZE_PROMPT = """You are a code-engineering memory extraction assistant. You will receive a conversation or context snippet (debugging session, development process, or project context).

Extract a single, concise, structured "experience entry" that can be reused in similar future tasks. Preserve the following information (omit any category if absent; do NOT fabricate):

1. **File paths / module names / class & function names** — keep original source identifiers (do NOT translate).
2. **Error messages / exception types / key logs / stack-fragment snippets** — keep verbatim.
3. **Commands / parameters / config items / environment variables** — keep verbatim.
4. **Root cause analysis** (1–2 sentences).
5. **Solution / fix steps / workaround** (1–2 sentences).
6. **Key code snippets** — API names, function signatures, important imports.
7. **Context tags** — domain (e.g., web, db, ml), error type (e.g., NPE, OOM, import), language.

Output: a single concise block. No markdown headings. No meta-commentary. Do not answer the original question."""


# Decide which of the three memory types the entry belongs to.
CLASSIFY_MEMORY_TYPE_PROMPT = """You are a code-engineering memory classification assistant. You will receive a single memory entry (extracted from a past conversation or task).

Classify the entry into exactly ONE of the following three types:

- **profile**: Stable, long-lived facts about a project or codebase that rarely change. Examples: "Project uses Python 3.12 + FastAPI", "Auth uses JWT with 24h TTL", "Database is PostgreSQL 15 with read replicas".

- **event**: Dynamic, time- or state-bound activities describing a debugging or development process. Examples: "Fixed NPE in auth.py by adding null check", "Migrated from REST to gRPC over 2 weeks", "CVE-2024-XXXX patched in v2.3.1".

- **record**: Detail-dense, source-grounded artifacts: function signatures, API usage, error traces, exact commands, configuration snippets. Examples: "Function foo(x: int) -> str in util.py raises on empty input", "Pinning requests==2.31.0 to fix SSL verify_failed".

Decision rules (apply in this order):
1. If the entry contains time/state verbs ("fixed", "migrated", "deployed", "refactored", "patched", "added", "removed", "upgraded") → **event**.
2. Else if the entry is a stable declarative fact about the project/codebase → **profile**.
3. Otherwise (specific code/config/snippet with details) → **record**.

Return JSON only: {"type": "profile"|"event"|"record", "reason": "<one sentence>"}"""


# Extract (attr, value) pairs for a profile memory (LeanMem §3.2 — done by LLM
# here, but the heuristic fallback in `domain/profile_extractor.py` covers
# LLM-down cases).
EXTRACT_PROFILE_PROMPT = """You are a code-engineering fact extractor. You will receive a memory entry classified as a "profile" (stable project/codebase fact).

Extract stable (attribute, value) pairs that describe long-lived project properties. Examples:
- ("language", "Python 3.12")
- ("framework", "FastAPI 0.115")
- ("auth", "JWT with 24h TTL")
- ("primary_database", "PostgreSQL 15")
- ("deployment", "Docker + Kubernetes on AWS")

Rules:
- Only extract facts that are stable and reusable across tasks.
- Use snake_case for attribute names.
- Keep values concise (one short phrase).
- 1–5 pairs total.

Output JSON only: {"pairs": [{"attr": "<name>", "value": "<value>"}, ...]}. No preamble."""


# Extract structured event fields (trigger / action / outcome / temporal_anchor).
EXTRACT_EVENT_PROMPT = """You are a code-engineering event log extractor. You will receive a memory entry describing a debugging or development activity.

Extract structured event fields:
1. **trigger**: what caused the activity (e.g., "user-reported bug", "scheduled upgrade", "CVE disclosure"). Empty string if not stated.
2. **action**: what was done (e.g., "added null check", "migrated endpoint", "refactored module").
3. **outcome**: result (e.g., "fixed in v1.2.3", "reduced latency by 40%", "still under investigation").
4. **temporal_anchor**: time reference if present (date, version, sprint, etc.); else "unspecified".

Output JSON only: {"trigger": "...", "action": "...", "outcome": "...", "temporal_anchor": "..."}. No preamble."""


# Extract gist + entities + keywords for a record memory.
EXTRACT_RECORD_PROMPT = """You are a code-engineering entity & keyword extractor. You will receive a memory entry classified as a "record" (detail-dense code/config/snippet).

Extract:
1. **gist**: a 1-sentence retrieval-oriented summary emphasizing what is unique about this record.
2. **entities**: code identifiers (function names, class names, module names, file paths, error types, library names, command names, version numbers).
3. **keywords**: 3–8 high-value terms that uniquely identify this record for retrieval.

Output JSON only: {"gist": "...", "entities": ["...", ...], "keywords": ["...", ...]}. No preamble."""


# ----------------------------------------------------------------------------
# Search pipeline
# ----------------------------------------------------------------------------

# Reduce the user query to a retrieval-friendly representation (strip noise,
# surface key identifiers).
ENHANCE_QUERY_PROMPT = """You are a code-engineering retrieval intent parser. You will be given a query (possibly from a code-engineering task).

Extract the following:
1. **Core intent** (1 sentence).
2. **Key code identifiers** (module/function/class names, error messages, file paths, commands).
3. **Key domain terms** (ordered by importance).

Output: a single retrieval text only. Do NOT answer the original query. Optimize for keyword and semantic matching against a memory bank of past code-engineering experiences."""


# Decide which memory type a query most needs (Adaptive Evidence Composition
# from LeanMem §3.4).
CLASSIFY_QUERY_INTENT_PROMPT = """You are a code-engineering query intent classifier. Given a query, decide which memory type it most needs:

- **profile**: A question about stable project facts ("what language", "what framework", "what version").
- **event**: A question about past debugging/fix experience ("how did I fix", "last time", "failed before", "上一次").
- **record**: A question about specific code details ("function signature", "API usage", "exact error", "how to call").

Output JSON only: {"types": ["<type1>", "<type2>", "<type3>"], "weights": [<float>, <float>, <float>]}.
- Weights must correspond to types in order; sum should be approximately 1.0.
- Order types by relevance (most relevant first).
- Include all 3 types; weigh 0.0 if not relevant at all."""


# ----------------------------------------------------------------------------
# Prompts — ExperienceCard extraction, decision-utility, contradiction
# ----------------------------------------------------------------------------

# Combined extraction: one LLM call per transcript segment produces the full
# ExperienceCard (replaces the older 3-call summarize→classify→extract chain).
# Grounded in the segment text + task context; never fabricates.
EXTRACT_EXPERIENCE_CARD_PROMPT = """You are a code-engineering experience extractor. You receive a transcript segment from a coding-agent session (debugging / development / refactor) plus the task context (repo, language, files touched).

Extract ONE reusable ExperienceCard. Preserve original source identifiers (file paths, function/class names, error messages, commands) VERBATIM — do NOT translate or paraphrase them. Omit a field by leaving it empty string / empty array; do NOT fabricate.

Fields:
- memory_type: "profile" (stable project fact) | "event" (a debug/fix/migrate action happened) | "record" (detail-dense code/config artifact).
- trigger: the symptom / condition / input that started the work (1 sentence).
- root_cause: the underlying cause discovered (1 sentence). Empty if not diagnosed.
- action_hint: the reusable fix / action / approach (1-2 sentences, imperative).
- files_symbols: array of file paths and function/class/module names touched or relevant.
- failure_if_ignored: what breaks or what mistake repeats if this is ignored (1 sentence). Empty if not applicable.
- scope: "{repo}/{language}" plus scenario tag if any (e.g. "django/python#admin-templates").
- temporal_anchor: version / commit / date if present; else "".
- gist: 1-sentence retrieval summary emphasizing what is unique.
- entities: array of code identifiers (function/class/module names, error types, library names).
- keywords: 3-8 high-value retrieval terms.
- pairs: only for profile type — array of {"attr": "...", "value": "..."} stable facts; else [].

Decision rules for memory_type (apply in order):
1. If the segment describes an action that happened (fixed/migrated/refactored/patched/added/removed) → "event".
2. Else if it states a stable, long-lived project/codebase fact → "profile".
3. Otherwise (specific code/config/error detail) → "record".

Output JSON ONLY matching this shape:
{"memory_type":"...","trigger":"...","root_cause":"...","action_hint":"...","files_symbols":["..."],"failure_if_ignored":"...","scope":"...","temporal_anchor":"...","gist":"...","entities":["..."],"keywords":["..."],"pairs":[{"attr":"...","value":"..."}]}"""


# Stage-2 decision-utility scoring (CICL). Given a query (intent + key terms)
# and a candidate card, score how much this memory would CHANGE the next action.
# Returns a 0-1 utility score; used as the rerank key.
DECISION_UTILITY_PROMPT = """You are a code-engineering decision-utility scorer. Given a task query and a candidate past-experience memory, estimate how useful this memory is for changing the agent's NEXT action toward solving the task.

Score each axis 0.0-1.0:
- action_relevance: does the action_hint directly apply to the query's task?
- failure_match: does failure_if_ignored match a risk the query is facing?
- scope_match: does the memory's scope (repo/language/scenario) match the query?
- recency: 1.0 if very recent, decaying to 0.0 for stale (use temporal_anchor).

Output JSON ONLY: {"action_relevance":0.0,"failure_match":0.0,"scope_match":0.0,"recency":0.0,"reason":"<one short sentence>"}

Rules:
- A memory that merely mentions the same keyword but offers no actionable hint scores low.
- A memory whose action_hint maps to the query's suspected root cause scores high.
- If you cannot tell, score 0.3 (neutral), not 0."""


# Revision: decide the relationship between a new candidate card and an
# existing near-duplicate (OpenAI Dreaming V3 supports/contradicts/refines).
RELATION_JUDGE_PROMPT = """You are a memory-relation judge. You receive a NEW candidate experience and an EXISTING stored experience. Decide their relationship:

- "duplicate": same experience, no new information.
- "supports": new adds minor corroborating detail; merge into existing.
- "refines": new adds a correction, extension, or more precise root_cause/action; merge (new fields override).
- "contradicts": new directly contradicts the existing root_cause or action_hint; the new should supersede the old (keep version chain).
- "unrelated": despite lexical similarity, semantically distinct; keep both.

Output JSON ONLY: {"relation":"duplicate|supports|refines|contradicts|unrelated","reason":"<one sentence>","merged_fields":{"trigger":"...","root_cause":"...","action_hint":"...","failure_if_ignored":"...","scope":"..."}}

For supports/refines, populate merged_fields with the merged values (existing as base, new overriding where it refines). For duplicate/contradicts/unrelated, leave merged_fields empty {}."""
