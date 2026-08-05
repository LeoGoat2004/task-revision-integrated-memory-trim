"""OpenAI chat-completion client with retry.

Design:
- Single `OpenAI` client instance is created lazily and reused (the official
  client is thread-safe).
- All calls use exponential backoff with jitter on transient errors
  (429, 5xx, network). Permanent errors (4xx other than 429) are surfaced
  immediately.
- When `OPENAI_API_KEY` is absent, the client returns graceful fallbacks
  (empty string / default values) so the service layer can stay online for
  offline development and tests.
"""
from __future__ import annotations

import json
import logging
import random
import threading
import time
from typing import Any

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from ..config import get_settings


logger = logging.getLogger(__name__)


# Retry policy
_MAX_RETRIES = 3
_INITIAL_BACKOFF_SECS = 0.5
_BACKOFF_FACTOR = 2.0
_JITTER_FRACTION = 0.2

# Seed the global RNG with the project seed so backoff jitter is reproducible
# across runs (per project convention: seed = 20234150).
from ..domain.params import RANDOM_SEED
random.seed(RANDOM_SEED)


_client: OpenAI | None = None


# ---------------------------------------------------------------------------
# Per-request circuit breaker (Claude Code lesson: a flapping LLM can death-
# loop and burn quota. After `llm_failure_threshold` consecutive failures the
# breaker trips and short-circuits subsequent calls to empty fallbacks for the
# rest of the request, so the pipeline degrades gracefully instead of looping).
# ---------------------------------------------------------------------------

class _Breaker:
    """Thread-local circuit breaker state."""

    def __init__(self) -> None:
        self._local = threading.local()

    @property
    def _state(self) -> dict[str, int]:
        st = getattr(self._local, "state", None)
        if st is None:
            st = {"failures": 0, "tripped": 0}
            self._local.state = st
        return st

    def reset(self) -> None:
        self._local.state = {"failures": 0, "tripped": 0}

    def record_success(self) -> None:
        st = self._state
        st["failures"] = 0

    def record_failure(self, threshold: int) -> None:
        st = self._state
        st["failures"] += 1
        if st["failures"] >= threshold:
            st["tripped"] = 1
            logger.warning(
                "LLM circuit breaker tripped after %d consecutive failures", st["failures"]
            )

    def is_tripped(self) -> bool:
        return bool(self._state["tripped"])


_breaker = _Breaker()


def reset_breaker() -> None:
    """Reset the per-request breaker (call at the start of each /add or /search)."""
    _breaker.reset()


def breaker_tripped() -> bool:
    """Whether the LLM circuit breaker has tripped (calls should short-circuit)."""
    return _breaker.is_tripped()


def _get_client() -> OpenAI | None:
    """Return the process-wide OpenAI client, or None if no API key."""
    global _client
    settings = get_settings()
    if not settings.openai_api_key:
        return None
    if _client is None:
        _client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
        )
    return _client


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter."""
    base = _INITIAL_BACKOFF_SECS * (_BACKOFF_FACTOR ** attempt)
    jitter = base * _JITTER_FRACTION * random.random()
    return base + jitter


# Track whether the API rejected the enable_thinking param. Once rejected,
# we stop sending it on subsequent calls — self-healing so the code works
# with ANY OpenAI-compatible model (Qwen3 needs it; GPT/Claude/Doubao ignore
# or reject it, and we gracefully adapt).
_thinking_param_rejected: bool = False


def _call_with_retry(payload: dict[str, Any]) -> str:
    """Call chat.completions.create with retry; return the assistant text or ''."""
    global _thinking_param_rejected
    client = _get_client()
    if client is None:
        return ""
    # Circuit breaker: if tripped, short-circuit to avoid death-looping.
    if _breaker.is_tripped():
        logger.debug("LLM call short-circuited (breaker tripped)")
        return ""
    settings = get_settings()
    # Qwen3 / thinking-mode models: pass enable_thinking=False via extra_body so
    # the model doesn't burn the entire max_tokens budget on chain-of-thought
    # before emitting the answer. Non-Qwen APIs silently ignore this param; if
    # a strict API rejects it, we self-heal by dropping the param and retrying.
    if not settings.llm_enable_thinking and not _thinking_param_rejected:
        payload.setdefault("extra_body", {})["enable_thinking"] = False
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(**payload)
            choice = resp.choices[0]
            _breaker.record_success()
            return (choice.message.content or "").strip()
        except (RateLimitError, APITimeoutError, APIError) as exc:
            last_exc = exc
            status = getattr(exc, "status_code", None)
            err_str = str(exc).lower()
            # Self-heal: if the API rejected the enable_thinking param, drop it
            # and retry immediately (don't count this against the retry budget).
            if (
                status is not None and 400 <= status < 500 and status != 429
                and "enable_thinking" in err_str
                and not _thinking_param_rejected
            ):
                _thinking_param_rejected = True
                eb = payload.get("extra_body", {})
                eb.pop("enable_thinking", None)
                if not eb:
                    payload.pop("extra_body", None)
                logger.info("API rejected enable_thinking param; retrying without it")
                continue
            # 429 / 5xx → retry; 4xx other → give up.
            if status is not None and 400 <= status < 500 and status != 429:
                logger.warning("LLM call failed (non-retryable %s): %s", status, exc)
                _breaker.record_failure(settings.llm_failure_threshold)
                return ""
            if attempt < _MAX_RETRIES - 1:
                sleep_for = _backoff_seconds(attempt)
                logger.info("LLM retry %d after %.2fs: %s", attempt + 1, sleep_for, exc)
                time.sleep(sleep_for)
            else:
                logger.warning("LLM call exhausted retries: %s", exc)
                _breaker.record_failure(settings.llm_failure_threshold)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM call unexpected error: %s", exc)
            _breaker.record_failure(settings.llm_failure_threshold)
            return ""
    if last_exc is not None:
        logger.warning("LLM last exception: %s", last_exc)
    return ""


def chat_json(
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> dict[str, Any] | None:
    """Call the chat completion endpoint with JSON-output expectations.

    Returns the parsed JSON dict, or None if the call failed or the response
    was not valid JSON. When the API key is missing, returns None so callers
    can decide on a fallback strategy.
    """
    settings = get_settings()
    body = {
        "model": model or settings.llm_model,
        "max_tokens": max_tokens if max_tokens is not None else settings.llm_max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    text = _call_with_retry(body)
    if not text:
        return None
    # Strip ```json ... ``` fences if present (defensive).
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to recover the first {...} block.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def chat_text(
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> str:
    """Free-form chat completion. Returns '' on failure (missing key or error)."""
    settings = get_settings()
    body = {
        "model": model or settings.llm_model,
        "max_tokens": max_tokens if max_tokens is not None else settings.llm_max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    return _call_with_retry(body)


def client_available() -> bool:
    """Whether the OpenAI client is configured (API key present)."""
    return _get_client() is not None
