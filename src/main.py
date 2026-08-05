"""Process entry point.

Reads config, builds the FastAPI app, and starts uvicorn.

Run with:
    cd src && python -m main
or:
    cd src && PYTHONPATH=. uvicorn main:app
"""
from __future__ import annotations

import uvicorn

from app.api.app import create_app
from app.config import get_settings


# Eager-create the app so `uvicorn main:app` works.
app = create_app()


def main() -> None:
    """Run the API server with the configured host/port."""
    settings = get_settings()
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
