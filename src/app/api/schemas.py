"""Pydantic request/response schemas.

This module is the contract at the wire boundary. The internal model
(`domain.MemoryNote`) may evolve freely; these classes must stay
backward-compatible.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


MessageRole = Literal["user", "assistant"]


class AddMessage(BaseModel):
    role: MessageRole
    timestamp: Optional[int] = None  # Unix milliseconds; optional
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def _strip_content(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("content must be non-empty after stripping")
        return v


class AddRequest(BaseModel):
    request_id: str = Field(min_length=1)
    messages: list[AddMessage] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class AddResponse(BaseModel):
    success: bool = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    top_k: int = Field(ge=1)
    options: Optional[list[str]] = None


class SearchResultItem(BaseModel):
    id: str
    content: str
    score: Optional[float] = None
    created_at: Optional[str] = None


class SearchResponse(BaseModel):
    data: list[SearchResultItem]


class ErrorDetail(BaseModel):
    reason: str


class ErrorResponse(BaseModel):
    detail: ErrorDetail
