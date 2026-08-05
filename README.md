# Code Memory Agent

Agent Memory Challenge 2026 · 代码记忆榜 · 学术方法榜 · 平台部署路径。

## Docker 启动

```bash
docker compose up --build
```

服务启动在 `http://0.0.0.0:8000`，健康检查 `GET /health` 返回 2xx。

## API 接口

### 鉴权

`/add` 和 `/search` 需要 Memory System Key，三选一：

| 方式 | Header |
| --- | --- |
| X-Api-Key | `X-Api-Key: <key>` |
| Bearer | `Authorization: Bearer <key>` |
| Token | `Authorization: Token <key>` |

`/health` 无需鉴权。

### GET /health

返回 200，无响应体。

### POST /add

同步写入记忆，持久化完成返回 200。

**请求字段：**

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `request_id` | string | 是 | 请求唯一标识 |
| `messages` | array | 是 | 对话消息列表，至少 1 条 |
| `messages[].role` | string | 是 | `"user"` 或 `"assistant"` |
| `messages[].content` | string | 是 | 消息文本内容 |
| `messages[].timestamp` | int | 否 | Unix 毫秒时间戳 |
| `user_id` | string | 是 | 用户标识（隔离边界） |
| `session_id` | string | 是 | 会话标识 |

**请求示例：**

```json
{
  "request_id": "req-001",
  "messages": [
    {"role": "user", "content": "How to fix CORS error in Django?"},
    {"role": "assistant", "content": "Add django-cors-headers to MIDDLEWARE."}
  ],
  "user_id": "user-001",
  "session_id": "session-001"
}
```

**响应字段：**

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `success` | bool | 是否写入成功 |
| `request_id` | string | 请求标识 |
| `user_id` | string | 用户标识 |
| `session_id` | string | 会话标识 |

**响应示例：**

```json
{
  "success": true,
  "request_id": "req-001",
  "user_id": "user-001",
  "session_id": "session-001"
}
```

### POST /search

按 `user_id` 隔离检索，返回最相关的记忆。

**请求字段：**

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `query` | string | 是 | 检索查询 |
| `user_id` | string | 是 | 用户标识（隔离边界） |
| `top_k` | int | 是 | 返回数量；正式评测使用 100 |
| `options` | array | 否 | 选项列表 |

**请求示例：**

```json
{
  "query": "Django CORS cross-origin",
  "user_id": "user-001",
  "top_k": 5,
  "options": []
}
```

**响应字段：**

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `data` | array | 记忆列表 |
| `data[].id` | string | 记忆唯一标识 |
| `data[].content` | string | 记忆内容 |
| `data[].score` | float | 相关度分数 |
| `data[].created_at` | string | 创建时间（ISO-8601） |

**响应示例：**

```json
{
  "data": [
    {
      "id": "mem_a1b2c3d4e5f6",
      "content": "CORS fix: add django-cors-headers to MIDDLEWARE",
      "score": 0.85,
      "created_at": "2026-08-05T12:00:00+00:00"
    }
  ]
}
```

## 依赖配置

```bash
pip install -r requirements.txt
```

环境变量（完整见 `.env.example`）：

| 变量 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | 是 | — | LLM 提供商 API Key |
| `OPENAI_BASE_URL` | 否 | OpenAI 官方 | LLM 端点 |
| `LLM_MODEL` | 否 | `gpt-4o-mini` | Add/Search 使用的模型 |
| `LLM_ENABLE_THINKING` | 否 | `false` | 是否开启 thinking 模式 |
| `LLM_MAX_TOKENS` | 否 | `2048` | 单次 LLM 调用最大输出 token |
| `EMBEDDING_API_KEY` | 否 | 回退到 `OPENAI_API_KEY` | Embedding 提供商 Key |
| `EMBEDDING_BASE_URL` | 否 | 回退到 `OPENAI_BASE_URL` | Embedding 端点 |
| `EMBEDDING_MODEL` | 否 | `text-embedding-3-small` | Embedding 模型 |
| `EMBEDDING_DIM` | 否 | `1536` | Embedding 维度 |
| `USE_LLM_ON_ADD` | 否 | `true` | Add 阶段是否使用 LLM |
| `USE_LLM_ON_SEARCH` | 否 | `true` | Search 阶段是否使用 LLM |
| `MEMORY_SYSTEM_KEY` | 否 | `dev-memory-system-key` | 接口鉴权 Key |
| `HOST` | 否 | `0.0.0.0` | 服务监听地址 |
| `PORT` | 否 | `8000` | 服务监听端口 |

当 `USE_LLM_ON_ADD=true` 或 `USE_LLM_ON_SEARCH=true` 时，必须提供 `OPENAI_API_KEY`。

## 运行步骤

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY

# 2. 构建并启动
docker compose up --build

# 3. 验证服务
curl http://localhost:8000/health
python src/scripts/local_smoke.py
```

本地运行（不用 Docker）：

```bash
pip install -r requirements.txt
cp .env.example .env  # 编辑填入 OPENAI_API_KEY
cd src && python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## 原始工作引用

| 来源 | 链接 | 借鉴点 |
| --- | --- | --- |
| SWEContextBench | https://github.com/jiayuanz3/SWEContextBench | 评测数据集与任务口径 |
| Agent Memory Leaderboard | https://agentmemories.ai/api-guide | Add/Search 契约 |
| LeanMem | arXiv:2608.03463 | 3 类记忆 + 查询自适应 |
| CICL | arXiv:2606.08151 | 决策效用重排 |
| GEM | arXiv:2605.26252 | ingestion/revision |
| ContextSniper | arXiv:2607.01916 | 多粒度返回 |
| Aider | https://github.com/Aider-AI/aider | tree-sitter 符号图 + PageRank |

## 方法说明

本系统为原创实现，未直接复用上述原始工作的代码，仅借鉴其方法思路：

- Add 管道：transcript 经 `{role, content}` 线格式摄入，按任务边界分段，每段由 LLM 抽取经验卡（event / profile / record 三类），经去重/合并/ supersede 修订后落库。
- Search 管道：BM25 + dense 混合召回，符号图 PageRank 扩展候选，决策效用重排，意图过滤后返回。
