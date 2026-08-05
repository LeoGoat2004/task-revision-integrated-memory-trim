"""POST /add — write memories."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..auth import require_memory_system_key
from ..schemas import AddRequest, AddResponse
from ...services import add_service


logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/add", response_model=AddResponse)
async def add(
    req: AddRequest,
    _: None = Depends(require_memory_system_key),
) -> AddResponse:
    try:
        written = add_service.add(
            user_id=req.user_id,
            session_id=req.session_id,
            request_id=req.request_id,
            messages=req.messages,
        )
        logger.info(
            "add user_id=%s request_id=%s written=%d",
            req.user_id, req.request_id, written,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("add failed for user_id=%s", req.user_id)
        raise HTTPException(status_code=500, detail={"reason": f"add failed: {exc}"})

    return AddResponse(
        success=True,
        request_id=req.request_id,
        user_id=req.user_id,
        session_id=req.session_id,
    )
