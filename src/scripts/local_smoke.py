"""本地 smoke 验证：Add → Search 链路是否正常。

用法：
    python src/scripts/local_smoke.py
    或设置环境变量 SMOKE_URL 指向远端服务。

覆盖关键能力：
  - /health 无鉴权可达
  - /add 接收多轮 user/assistant transcript，落库 ≥1 条
  - /search 返回结果
  - /search 用户隔离
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.config import get_settings


def main() -> int:
    settings = get_settings()
    base = os.environ.get("SMOKE_URL", f"http://127.0.0.1:{settings.port}")
    headers = {"X-Api-Key": settings.memory_system_key}

    with httpx.Client(base_url=base, timeout=30.0) as client:
        # /health（无需鉴权）
        r = client.get("/health")
        print(f"GET /health -> {r.status_code}")
        if r.status_code != 200:
            print("health failed")
            return 1

        # /add — transcript (user + assistant)
        add_body = {
            "request_id": "smoke:add:0",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "How do I fix the CORS error on /api/login in Django? "
                        "Browser blocks the cross-origin request."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Add django-cors-headers to MIDDLEWARE and set "
                        "CORS_ALLOWED_ORIGINS=['http://localhost:3000']. "
                        "Root cause: missing CORS middleware."
                    ),
                },
            ],
            "user_id": "smoke:user:0",
            "session_id": "smoke:session:0",
        }
        r = client.post("/add", json=add_body, headers=headers)
        print(f"POST /add -> {r.status_code} {r.json()}")
        if r.status_code != 200 or not r.json().get("success"):
            print("add failed")
            return 1

        # /search
        search_body = {
            "query": "Django CORS cross-origin login",
            "user_id": "smoke:user:0",
            "top_k": 5,
        }
        r = client.post("/search", json=search_body, headers=headers)
        print(f"POST /search -> {r.status_code}")
        data = r.json().get("data", [])
        print(f"  results: {len(data)} items")
        for it in data[:3]:
            print(
                f"  - {it.get('id')} score={it.get('score')} :: "
                f"{(it.get('content') or '')[:70]}..."
            )
        if r.status_code != 200 or not data:
            print("search failed")
            return 1

        # /search — user isolation
        r = client.post(
            "/search",
            json={"query": "CORS", "user_id": "smoke:user:other", "top_k": 5},
            headers=headers,
        )
        assert r.status_code == 200 and r.json().get("data") == [], "user isolation broken"
        print("  user isolation OK")

    print("smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
