"""GET /health — liveness probe."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response


router = APIRouter()


@router.get("/health", include_in_schema=False)
async def health() -> Response:
    """Return 200 OK if the process is alive. Used by orchestrators."""
    return Response(status_code=200)
