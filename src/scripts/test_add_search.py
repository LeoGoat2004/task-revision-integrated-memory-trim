"""Focused add+search test with real LLM + embedding. Prints full payloads."""
import sys, os, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import httpx
from app.config import get_settings

s = get_settings()
base = os.environ.get("SMOKE_URL", f"http://127.0.0.1:{s.port}")
headers = {"X-Api-Key": s.memory_system_key}

# Three realistic coding memories covering event / profile / record types.
MEMORIES = [
    {
        "user_id": "test:real:a",
        "session_id": "test:real:session",
        "request_id": "test:real:1",
        "messages": [
            {"role": "user", "content": "I keep getting AttributeError: 'NoneType' object has no attribute 'get' in django/db/models/query.py when chaining .filter().exists(). How do I fix this?"},
            {"role": "assistant", "content": "Root cause: the queryset is None because a previous .first() returned None and you chained .filter() on it. Fix: check for None before chaining, or use .filter(...).exists() directly without .first(). In django/db/models/query.py line 380, QuerySet.filter returns an empty queryset, not None — the bug is in the calling code pattern queryset.first().filter() which raises AttributeError."},
        ],
    },
    {
        "user_id": "test:real:a",
        "session_id": "test:real:session",
        "request_id": "test:real:2",
        "messages": [
            {"role": "user", "content": "What's the project convention for random seed and test fixtures?"},
            {"role": "assistant", "content": "Project rule: random seed must be set to 20234150 in all experiments. Old training results must be deleted before retraining. Parameters should be placed at the beginning of the file, not hard-coded. Use relative paths in code."},
        ],
    },
    {
        "user_id": "test:real:a",
        "session_id": "test:real:session",
        "request_id": "test:real:3",
        "messages": [
            {"role": "user", "content": "How to configure CORS for the FastAPI backend on port 8000?"},
            {"role": "assistant", "content": "Add CORSMiddleware to app: from starlette.middleware.cors import CORSMiddleware; app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:3000'], allow_methods=['*'], allow_headers=['*']). Root cause of CORS errors: missing middleware or wrong allow_origins. config.py line 45 has the settings."},
        ],
    },
]

# Search queries — each should match a different memory above.
QUERIES = [
    ("AttributeError NoneType queryset filter", "test:real:a"),
    ("random seed convention project rule", "test:real:a"),
    ("CORS FastAPI middleware configuration", "test:real:a"),
    ("django None filter chain bug", "test:real:a"),
]

def main():
    with httpx.Client(base_url=base, timeout=60.0) as c:
        # Health
        r = c.get("/health")
        print(f"GET /health -> {r.status_code}")
        assert r.status_code == 200

        # Add 3 memories
        for m in MEMORIES:
            r = c.post("/add", json=m, headers=headers)
            data = r.json()
            print(f"POST /add [{m['request_id']}] -> {r.status_code} success={data.get('success')}")
            assert r.status_code == 200 and data.get("success")

        # Search
        print("\n--- SEARCH ---")
        for q, uid in QUERIES:
            r = c.post("/search", json={
                "query": q, "user_id": uid, "top_k": 3,
            }, headers=headers)
            assert r.status_code == 200
            items = r.json()["data"]
            print(f"\nQ: {q}")
            for it in items[:2]:
                print(f"  score={it.get('score', 0):.4f}")
                print(f"  {it['content'][:200]}")

        # User isolation
        r = c.post("/search", json={
            "query": "CORS", "user_id": "test:real:other", "top_k": 5,
        }, headers=headers)
        assert r.json()["data"] == []
        print("\nuser isolation: OK")

        print("\nALL TESTS PASSED")

if __name__ == "__main__":
    main()
