"""FastAPI application factory.

The factory pattern lets tests inject a custom app (e.g., with overridden
dependencies). Production uses `create_app()`.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI

from ..config import get_settings
from ..infrastructure.sqlite import init_db
from ..utils.logging import setup_logging
from .routes import add as add_route
from .routes import health as health_route
from .routes import search as search_route


def create_app() -> FastAPI:
    """Build and return the FastAPI app."""
    settings = get_settings()
    setup_logging()
    init_db()

    app = FastAPI(
        title="Code Memory Agent",
        description=(
            "Add / Search API for the Agent Memory Challenge 2026 code-memory track."
        ),
    )
    app.include_router(add_route.router)
    app.include_router(search_route.router)
    app.include_router(health_route.router)

    logging.getLogger(__name__).info(
        "App started: llm=%s embed=%s dim=%d top_k_default=%d",
        settings.llm_model,
        settings.embedding_model,
        settings.embedding_dim,
        settings.top_k,
    )
    return app
