from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # ── LLM Provider ──────────────────────────────────────────────────────────
    openrouter_api_key: str = Field(..., env="OPENROUTER_API_KEY")
    # Comma-separated pool of API keys for rotation (e.g. key1,key2,key3).
    # When set, a random key is picked on every LLM call, distributing load
    # across multiple accounts to avoid per-account rate limits.
    # Falls back to OPENROUTER_API_KEY when empty.
    openrouter_api_keys: str = Field(default="", env="OPENROUTER_API_KEYS")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", env="OPENROUTER_BASE_URL"
    )
    openrouter_model: str = Field(
        default="openrouter/free", env="OPENROUTER_MODEL"
    )
    # Comma-separated fallback models tried in order on rate-limit / bad-model errors
    openrouter_fallback_models: str = Field(
        default="nvidia/nemotron-3-ultra-550b-a55b:free,meta-llama/llama-3.3-70b-instruct:free,meta-llama/llama-3.2-3b-instruct:free",
        env="OPENROUTER_FALLBACK_MODELS",
    )
    openrouter_router_model: str = Field(
        default="meta-llama/llama-3.2-3b-instruct:free",
        env="OPENROUTER_ROUTER_MODEL",
    )
    openrouter_router_fallback_models: str = Field(
        default="",
        env="OPENROUTER_ROUTER_FALLBACK_MODELS",
    )

    # ── API Auth ───────────────────────────────────────────────────────────────
    amop_api_key: str = Field(..., env="AMOP_API_KEY")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = Field(default="qdrant", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_collection: str = Field(default="work_orders", env="QDRANT_COLLECTION")

    # ── SQLite ────────────────────────────────────────────────────────────────
    sqlite_path: str = Field(default="/app/data/amop.db", env="SQLITE_PATH")

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=384, env="EMBEDDING_DIM")

    # ── Model Paths ───────────────────────────────────────────────────────────
    model_path: str = Field(default="/app/models/failure_model.joblib", env="MODEL_PATH")
    bm25_index_path: str = Field(default="/app/models/bm25_index.pkl", env="BM25_INDEX_PATH")

    # ── App ───────────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    app_env: str = Field(default="development", env="APP_ENV")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance — loaded once at startup.
    Use as a FastAPI dependency: settings: Settings = Depends(get_settings)
    """
    return Settings()
