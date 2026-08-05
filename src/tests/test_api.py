"""Smoke tests for the live service: full HTTP pipeline with real FastAPI."""
from __future__ import annotations


def test_health_no_auth(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_add_search_smoke(client, headers):
    add = client.post(
        "/add",
        json={
            "request_id": "smoke:add:0",
            "messages": [
                {"role": "user", "content": "Fixed NPE in auth.py"},
                {"role": "assistant", "content": "Added null check."},
            ],
            "user_id": "smoke:user:0",
            "session_id": "smoke:session:0",
        },
        headers=headers,
    )
    assert add.status_code == 200
    assert add.json()["success"] is True

    res = client.post(
        "/search",
        json={
            "query": "NPE auth",
            "user_id": "smoke:user:0",
            "top_k": 5,
        },
        headers=headers,
    )
    assert res.status_code == 200
    items = res.json()["data"]
    assert len(items) >= 1
    # Each item has the required fields.
    for item in items:
        assert {"id", "content", "score", "created_at"} <= set(item.keys())
