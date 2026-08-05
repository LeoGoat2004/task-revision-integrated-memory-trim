"""Integration tests for the services layer.

These tests use the FastAPI test client and exercise the end-to-end Add and
Search pipelines (no LLM calls; the `USE_LLM_ON_*` env flags are false).
"""
from __future__ import annotations


def test_add_then_search_returns_item_with_event_type(client, headers):
    """Plain Add+Search contract: after adding a note, search returns it.

    Heuristic classifier: "Fixed NPE in auth.py" contains the event verb
    "Fixed" → classified as `event` (a debug/fix action happened), not the
    no-LLM fallback of `record`. This is the intended smarter behaviour.
    """
    add = client.post(
        "/add",
        json={
            "request_id": "test:services:1",
            "messages": [{"role": "user", "content": "Fixed NPE in auth.py"}],
            "user_id": "test:services:user",
            "session_id": "test:services:session",
        },
        headers=headers,
    )
    assert add.status_code == 200
    assert add.json()["success"] is True

    res = client.post(
        "/search",
        json={
            "query": "NPE auth",
            "user_id": "test:services:user",
            "top_k": 1,
        },
        headers=headers,
    )
    assert res.status_code == 200
    items = res.json()["data"]
    assert len(items) == 1
    assert "NPE" in items[0]["content"]


def test_add_validates_pydantic_rejects_empty_content(client, headers):
    """The wire contract rejects empty message content."""
    r = client.post(
        "/add",
        json={
            "request_id": "test:services:bad",
            "messages": [{"role": "user", "content": "   "}],
            "user_id": "test:services:user",
            "session_id": "test:services:session",
        },
        headers=headers,
    )
    assert r.status_code == 422


def test_add_validates_pydantic_rejects_missing_user_id(client, headers):
    r = client.post(
        "/add",
        json={
            "request_id": "test:services:bad",
            "messages": [{"role": "user", "content": "ok"}],
            "session_id": "test:services:session",
        },
        headers=headers,
    )
    assert r.status_code == 422


def test_search_rejects_zero_top_k(client, headers):
    r = client.post(
        "/search",
        json={"query": "x", "user_id": "test:services:user", "top_k": 0},
        headers=headers,
    )
    assert r.status_code == 422


def test_search_returns_empty_for_empty_user(client, headers):
    r = client.post(
        "/search",
        json={"query": "any", "user_id": "test:services:nobody", "top_k": 5},
        headers=headers,
    )
    assert r.status_code == 200
    assert r.json()["data"] == []


def test_add_chunks_when_long_then_each_chunk_persisted(client, headers):
    """Verify the chunker path produces ≥ 1 memory per chunk."""
    # 45 messages — exceeds MAX_MESSAGES_PER_CHUNK = 20.
    msgs = [{"role": "user", "content": f"debug log line {i}"} for i in range(45)]
    r = client.post(
        "/add",
        json={
            "request_id": "test:services:chunk",
            "messages": msgs,
            "user_id": "test:services:chunk",
            "session_id": "test:services:session",
        },
        headers=headers,
    )
    assert r.status_code == 200
    assert r.json()["success"] is True

    # Search should return some memories (each chunk got persisted).
    res = client.post(
        "/search",
        json={
            "query": "debug log",
            "user_id": "test:services:chunk",
            "top_k": 50,
        },
        headers=headers,
    )
    assert res.status_code == 200
    assert len(res.json()["data"]) >= 1


def test_user_isolation(client, headers):
    """Search for user A must not return user B's memory."""
    client.post(
        "/add",
        json={
            "request_id": "test:services:iso_a",
            "messages": [{"role": "user", "content": "private data for A"}],
            "user_id": "test:services:user_A",
            "session_id": "test:services:session",
        },
        headers=headers,
    )
    r = client.post(
        "/search",
        json={
            "query": "private",
            "user_id": "test:services:user_B",
            "top_k": 10,
        },
        headers=headers,
    )
    assert r.status_code == 200
    assert all("private" not in (it["content"] or "") for it in r.json()["data"])


def test_auth_fails_closed_without_key(client, headers):
    """Without the X-Api-Key header, POST /add must return 401."""
    r = client.post(
        "/add",
        json={
            "request_id": "test:services:noauth",
            "messages": [{"role": "user", "content": "ok"}],
            "user_id": "test:services:user",
            "session_id": "test:services:session",
        },
    )
    assert r.status_code == 401


def test_auth_bearer_works(client, bearer_headers):
    r = client.post(
        "/add",
        json={
            "request_id": "test:services:bearer",
            "messages": [{"role": "user", "content": "auth via bearer"}],
            "user_id": "test:services:bearer_user",
            "session_id": "test:services:session",
        },
        headers=bearer_headers,
    )
    assert r.status_code == 200


def test_add_dedup_skips_duplicate_content(client, headers):
    """Submitting the same content twice yields only one persisted memory."""
    user = "test:services:dedup"
    body = {
        "request_id": "test:services:dedup",
        "messages": [{"role": "user", "content": "added CORS headers for localhost dev"}],
        "user_id": user,
        "session_id": "test:services:session",
    }
    r1 = client.post("/add", json=body, headers=headers)
    r2 = client.post("/add", json=body, headers=headers)
    assert r1.status_code == 200
    assert r2.status_code == 200
    # Search by user: should see only one record.
    res = client.post(
        "/search",
        json={"query": "CORS", "user_id": user, "top_k": 10},
        headers=headers,
    )
    assert len(res.json()["data"]) == 1


# ---------------------------------------------------------------------------
# Revision merge, multi-message transcript
# ---------------------------------------------------------------------------


def test_revision_merges_near_duplicate_instead_of_appending(client, headers):
    """A second Add with overlapping-but-not-identical content merges (1 memory)."""
    user = "test:services:merge"
    base = {
        "user_id": user,
        "session_id": "test:services:session",
    }
    client.post(
        "/add",
        json={
            **base,
            "request_id": "test:services:merge:1",
            "messages": [{"role": "user", "content": "Fixed NPE in auth.py by adding null check on user_id"}],
        },
        headers=headers,
    )
    # A refinement: shares the root identifiers (NPE, auth.py, user_id) but adds
    # a new detail (session_id guard). Token overlap lands in the "supports"
    # band of the zero-LLM relation judge → MERGE, not SKIP (exact dup) and not
    # INSERT (unrelated). Result: one merged memory, version bumped.
    client.post(
        "/add",
        json={
            **base,
            "request_id": "test:services:merge:2",
            "messages": [{"role": "user", "content": "Fixed NPE in auth.py null check on user_id now also guards session_id"}],
        },
        headers=headers,
    )
    res = client.post(
        "/search",
        json={"query": "NPE auth", "user_id": user, "top_k": 10},
        headers=headers,
    )
    # One merged memory, not two.
    assert len(res.json()["data"]) == 1


def test_add_multi_message_transcript_persists_experience(client, headers):
    """A user/assistant transcript is parsed and yields a retrievable memory."""
    user = "test:services:multi"
    client.post(
        "/add",
        json={
            "request_id": "test:services:multi",
            "messages": [
                {"role": "user", "content": "How do I fix the CORS error on /api/login?"},
                {"role": "assistant", "content": "Add corsheaders to MIDDLEWARE and set CORS_ALLOWED_ORIGINS."},
            ],
            "user_id": user,
            "session_id": "test:services:session",
        },
        headers=headers,
    )
    res = client.post(
        "/search",
        json={"query": "CORS login", "user_id": user, "top_k": 5},
        headers=headers,
    )
    assert res.status_code == 200
    items = res.json()["data"]
    assert len(items) >= 1
    assert "CORS" in items[0]["content"]
