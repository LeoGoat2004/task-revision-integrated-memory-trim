"""Memory System Key authentication.

Supports three authentication schemes (in order of preference):
  1. `X-Api-Key: <token>` (header)
  2. `Authorization: Bearer <token>`
  3. `Authorization: Token <token>`

The configured key is compared in constant time. When the configured key is
empty, authentication fails closed (every request is rejected).
"""
from __future__ import annotations

import hmac
from typing import Optional

from fastapi import Header, HTTPException, status

from ..config import get_settings


def _extract(authorization: Optional[str], x_api_key: Optional[str]) -> Optional[str]:
    if x_api_key:
        return x_api_key.strip()
    if authorization:
        parts = authorization.strip().split(None, 1)
        if len(parts) == 2 and parts[0].lower() in ("bearer", "token"):
            return parts[1].strip()
    return None


async def require_memory_system_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-Api-Key"),
) -> None:
    """FastAPI dependency that enforces the Memory System Key.

    Raises HTTP 401 with the platform's `{"detail": {"reason": "..."}}`
    shape on failure.
    """
    settings = get_settings()
    expected = settings.memory_system_key
    provided = _extract(authorization, x_api_key)
    if not expected:
        # Fail closed: no configured key means no access.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"reason": "memory system key not configured"},
        )
    if not provided:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"reason": "missing memory system key"},
        )
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"reason": "invalid memory system key"},
        )
