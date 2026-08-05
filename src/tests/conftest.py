"""Pytest configuration.

Sets up an isolated temporary database, disables LLM-call paths (no real
network calls during tests), and provides a fresh FastAPI test client.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add `src` to sys.path so `app.*` imports resolve.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Configure environment BEFORE any app module imports.
TEST_DB = _ROOT / "tests" / "tmp.db"
TEST_KEY = "test-memory-key"

os.environ["MEMORY_SYSTEM_KEY"] = TEST_KEY
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ["DB_PATH"] = str(TEST_DB)
os.environ["USE_LLM_ON_ADD"] = "false"
os.environ["USE_LLM_ON_SEARCH"] = "false"

# Wipe any stale DB.
if TEST_DB.exists():
    TEST_DB.unlink()


# Clear the lru_cache for get_settings so env vars are read fresh.
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.api.app import create_app  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.infrastructure.sqlite import init_db  # noqa: E402


get_settings.cache_clear()


@pytest.fixture(scope="session")
def client() -> TestClient:
    init_db()
    return TestClient(create_app())


@pytest.fixture(scope="session")
def headers() -> dict[str, str]:
    return {"X-Api-Key": TEST_KEY}


@pytest.fixture(scope="session")
def bearer_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TEST_KEY}"}
