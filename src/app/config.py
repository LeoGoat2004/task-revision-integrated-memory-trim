"""Application settings backed by environment variables / `.env`.

Resolution order for any setting:
  1. Process environment variables (e.g., `OPENAI_API_KEY`).
  2. The `.env` file at the repository root.
  3. The default declared on the field.

All tunable algorithm parameters live here as named fields (overridable via
env) and are projected onto the pure `domain.params.DomainParams` via
`get_domain_params()`. Domain modules never import this file — they receive
`DomainParams` from the service layer, keeping the domain IO-free.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from .domain.params import DomainParams, RANDOM_SEED


BASE_DIR = Path(__file__).resolve().parent.parent  # .../src
REPO_ROOT = BASE_DIR.parent  # .../code-memory-api


class Settings(BaseSettings):
    # === GLOBAL SEED (project convention: 20234150) ========================
    # Exposed via env so CI can pin it; defaults to the canonical project seed.
    random_seed: int = RANDOM_SEED

    # === Server ============================================================
    host: str = "0.0.0.0"
    port: int = 8000

    # === OpenAI / models ===================================================
    # LLM (chat completions) — uses OPENAI_API_KEY / OPENAI_BASE_URL.
    openai_api_key: str = ""
    openai_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"
    # Qwen3 / DeepSeek-R1 etc. have a "thinking" mode that emits reasoning
    # tokens BEFORE the answer, consuming the max_tokens budget and producing
    # empty output. Set to false to pass enable_thinking=False (Qwen-specific,
    # silently ignored by non-Qwen APIs). Set true if your model benefits from
    # chain-of-thought AND you raise llm_max_tokens accordingly.
    llm_enable_thinking: bool = False
    # Max output tokens per LLM call. 2048 is enough for ExperienceCard JSON
    # with thinking disabled; raise to 4096+ if thinking is enabled.
    llm_max_tokens: int = 2048
    # Embeddings — may use a DIFFERENT provider than the LLM. When the
    # EMBEDDING_* vars are unset, the embedder falls back to the OPENAI_*
    # values above (single-provider case still works with one key).
    embedding_api_key: str = ""
    embedding_base_url: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # === Auth (Memory System Key) ==========================================
    memory_system_key: str = "dev-memory-system-key"

    # === Storage ===========================================================
    db_path: str = str(BASE_DIR / "data" / "memories.db")

    # === Behavior flags ====================================================
    use_llm_on_add: bool = True
    use_llm_on_search: bool = True

    # === Retrieval (Stage 1) — mirrored into DomainParams ==================
    top_k: int = 100
    recall_multiplier: int = 5
    type_bonus: float = 0.3
    temporal_decay: float = 0.0077
    temporal_floor: float = 0.5

    # === Decision-utility rerank (Stage 2) =================================
    w_action_relevance: float = 0.35
    w_failure_match: float = 0.25
    w_scope_match: float = 0.20
    w_recency: float = 0.10
    w_type_weight: float = 0.10

    # === Chunker (task-boundary) ===========================================
    chunk_max_messages: int = 30
    chunk_max_words: int = 3000
    chunk_boundary_on_todo: bool = True

    # === Revision / dedup ==================================================
    near_dup_threshold: float = 0.85
    dedup_neighbor_cap: int = 100
    revision_neighbor_topk: int = 3
    revision_sim_threshold: float = 0.78

    # === Symbol graph ======================================================
    pagerank_edited_boost: float = 50.0
    pagerank_damping: float = 0.85
    pagerank_iterations: int = 40
    pagerank_top_n: int = 10
    symbol_expand_enabled: bool = True

    # === Context gate ======================================================
    default_granularity: str = "L1"
    l0_max_chars: int = 160

    # === LLM circuit breaker ===============================================
    llm_failure_threshold: int = 3
    llm_batch_size: int = 5

    # === Transcript parsing ================================================
    transcript_tool_result_cap: int = 1200
    transcript_max_segment_chars: int = 8000

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def to_domain_params(self) -> DomainParams:
        """Project the env-backed settings onto the pure `DomainParams`."""
        return DomainParams(
            top_k=self.top_k,
            recall_multiplier=self.recall_multiplier,
            type_bonus=self.type_bonus,
            temporal_decay=self.temporal_decay,
            temporal_floor=self.temporal_floor,
            w_action_relevance=self.w_action_relevance,
            w_failure_match=self.w_failure_match,
            w_scope_match=self.w_scope_match,
            w_recency=self.w_recency,
            w_type_weight=self.w_type_weight,
            chunk_max_messages=self.chunk_max_messages,
            chunk_max_words=self.chunk_max_words,
            chunk_boundary_on_todo=self.chunk_boundary_on_todo,
            near_dup_threshold=self.near_dup_threshold,
            dedup_neighbor_cap=self.dedup_neighbor_cap,
            revision_neighbor_topk=self.revision_neighbor_topk,
            revision_sim_threshold=self.revision_sim_threshold,
            pagerank_edited_boost=self.pagerank_edited_boost,
            pagerank_damping=self.pagerank_damping,
            pagerank_iterations=self.pagerank_iterations,
            pagerank_top_n=self.pagerank_top_n,
            symbol_expand_enabled=self.symbol_expand_enabled,
            default_granularity=self.default_granularity,
            l0_max_chars=self.l0_max_chars,
            llm_failure_threshold=self.llm_failure_threshold,
            llm_batch_size=self.llm_batch_size,
            transcript_tool_result_cap=self.transcript_tool_result_cap,
            transcript_max_segment_chars=self.transcript_max_segment_chars,
        )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if (settings.use_llm_on_add or settings.use_llm_on_search) and not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required when USE_LLM_ON_ADD or USE_LLM_ON_SEARCH is enabled"
        )
    return settings


@lru_cache
def get_domain_params() -> DomainParams:
    """Convenience accessor used by the service layer."""
    return get_settings().to_domain_params()
