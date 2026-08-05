"""POST /search — retrieve memories."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..auth import require_memory_system_key
from ..schemas import SearchRequest, SearchResponse, SearchResultItem
from ...services import search_service


logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    _: None = Depends(require_memory_system_key),
) -> SearchResponse:
    try:
        results = search_service.search(
            user_id=req.user_id,
            query=req.query,
            top_k=req.top_k,
            options=req.options,
        )
        logger.info(
            "search user_id=%s top_k=%d returned=%d",
            req.user_id, req.top_k, len(results),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("search failed for user_id=%s", req.user_id)
        raise HTTPException(status_code=500, detail={"reason": f"search failed: {exc}"})

    items = [
        SearchResultItem(
            id=r.id,
            content=r.content,
            score=r.score,
            created_at=r.created_at,
        )
        for r in results
    ]
    return SearchResponse(data=items)
