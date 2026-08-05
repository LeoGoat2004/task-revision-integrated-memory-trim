"""Logging setup.

Uses stdlib `logging` with a consistent structured format. Avoids extra
dependencies (no `structlog`, no `loguru`) so the dependency surface stays
small.

Default level: INFO. Override via `LOG_LEVEL=DEBUG` env var.
"""
from __future__ import annotations

import logging
import os
import sys


_FORMAT = "%(asctime)s %(levelname)-7s %(name)s :: %(message)s"
_DATE_FMT = "%Y-%m-%dT%H:%M:%S%z"


def setup_logging(level: str | None = None) -> None:
    """Configure the root logger. Idempotent (safe to call multiple times)."""
    effective = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FMT))
        root.addHandler(handler)
    root.setLevel(getattr(logging, effective, logging.INFO))
    # Quiet down noisy HTTP libs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
