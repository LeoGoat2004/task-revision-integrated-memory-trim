"""OpenAI embedding client.

- Single shared client, reused across calls (OpenAI client is thread-safe).
- Supports a **separate** embedding provider from the LLM provider: when
  `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL` are set they take precedence;
  otherwise the embedder falls back to `OPENAI_API_KEY` / `OPENAI_BASE_URL`
  (single-provider case).
- When no key is available, returns zero vectors so the rest of the pipeline
  can still operate (BM25-only retrieval falls back gracefully).
- Embedding dimension is configurable but defaults to `text-embedding-3-small`'s
  1536.
"""
from __future__ import annotations

import logging
from typing import Sequence

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from ..config import get_settings


logger = logging.getLogger(__name__)


_MAX_RETRIES = 3


_client: OpenAI | None = None


def _get_client() -> OpenAI | None:
    global _client
    settings = get_settings()
    # Prefer embedding-specific credentials; fall back to the LLM (OPENAI_*)
    # credentials so a single-provider setup still works with one key.
    api_key = settings.embedding_api_key or settings.openai_api_key
    if not api_key:
        return None
    if _client is None:
        base_url = settings.embedding_base_url or settings.openai_base_url or None
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def _zero_vector(dim: int) -> list[float]:
    return [0.0] * dim


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns zero vectors on failure (no API key)."""
    settings = get_settings()
    dim = settings.embedding_dim
    if not texts:
        return []
    client = _get_client()
    if client is None:
        return [_zero_vector(dim) for _ in texts]
    out: list[list[float]] = []
    for text in texts:
        out.append(_embed_one(client, text, dim))
    return out


def _embed_one(client: OpenAI, text: str, dim: int) -> list[float]:
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.embeddings.create(model=settings_ef_model(), input=text)
            emb = list(resp.data[0].embedding)
            if len(emb) != dim:
                logger.warning("Embedding dim mismatch: got %d expected %d", len(emb), dim)
                return _zero_vector(dim)
            return emb
        except (RateLimitError, APITimeoutError, APIError) as exc:
            last_exc = exc
            status = getattr(exc, "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                logger.warning("Embed failed (non-retryable %s): %s", status, exc)
                return _zero_vector(dim)
            if attempt < _MAX_RETRIES - 1:
                import time
                time.sleep(0.5 * (2 ** attempt))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embed unexpected error: %s", exc)
            return _zero_vector(dim)
    if last_exc is not None:
        logger.warning("Embed exhausted retries: %s", last_exc)
    return _zero_vector(dim)


def settings_ef_model() -> str:
    """Helper: read the configured embedding model name."""
    return get_settings().embedding_model


def embedder_available() -> bool:
    """Whether the OpenAI embedding client is configured."""
    return _get_client() is not None
