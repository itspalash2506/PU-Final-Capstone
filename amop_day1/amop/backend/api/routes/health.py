from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.db.session import get_db
from backend.core.config import get_settings, Settings
from backend.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    """
    Checks all system dependencies and returns a status summary.
    Returns 200 even if degraded — callers check the 'status' field.
    """
    components: dict[str, str] = {}

    # ── SQLite ────────────────────────────────────────────────────────────────
    try:
        db.execute(text("SELECT 1"))
        components["sqlite"] = "ok"
    except Exception as e:
        logger.error("SQLite health check failed", extra={"error": str(e)})
        components["sqlite"] = "error"

    # ── Qdrant ────────────────────────────────────────────────────────────────
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=3)
        client.get_collections()
        components["qdrant"] = "ok"
    except Exception as e:
        logger.warning("Qdrant health check failed", extra={"error": str(e)})
        components["qdrant"] = "error"

    # ── OpenRouter LLM (lightweight ping) ─────────────────────────────────────
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as http:
            resp = await http.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            )
        components["openrouter_llm"] = "ok" if resp.status_code == 200 else "error"
    except Exception as e:
        logger.warning("OpenRouter health check failed", extra={"error": str(e)})
        components["openrouter_llm"] = "error"

    # ── Local embedding model ─────────────────────────────────────────────────
    try:
        # Check if model is cached — don't load it just for health check
        from sentence_transformers import SentenceTransformer
        import os
        cache_dir = os.path.expanduser("~/.cache/torch/sentence_transformers")
        model_cached = any(
            settings.embedding_model.lower().replace("/", "_") in d.lower()
            for d in os.listdir(cache_dir)
        ) if os.path.exists(cache_dir) else False
        components["embeddings_model"] = "ok" if model_cached else "not_loaded_yet"
    except Exception:
        components["embeddings_model"] = "unknown"

    # ── BM25 index ────────────────────────────────────────────────────────────
    try:
        import os
        components["bm25_index"] = "ok" if os.path.exists(settings.bm25_index_path) else "not_built_yet"
    except Exception:
        components["bm25_index"] = "error"

    # ── XGBoost prediction model ──────────────────────────────────────────────
    try:
        import os
        components["prediction_model"] = "ok" if os.path.exists(settings.model_path) else "not_trained_yet"
    except Exception:
        components["prediction_model"] = "error"

    # ── Cross-encoder reranker ────────────────────────────────────────────────
    # Check cache_info() — never trigger a load from a health check.
    # "ok" means startup eager-load succeeded; "not_loaded" means it failed or
    # hasn't run yet (reranking will fall back to RRF order in that state).
    try:
        from backend.rag.reranker import get_reranker
        components["reranker"] = "ok" if get_reranker.cache_info().currsize > 0 else "not_loaded"
    except Exception as e:
        logger.warning("Reranker health check failed", extra={"error": str(e)})
        components["reranker"] = "error"

    # ── Overall status ────────────────────────────────────────────────────────
    errors = [k for k, v in components.items() if v == "error"]
    if not errors:
        overall = "healthy"
    elif "sqlite" in errors or "qdrant" in errors:
        overall = "unhealthy"
    else:
        overall = "degraded"

    logger.info("Health check completed", extra={"status": overall, "components": components})

    return {
        "status": overall,
        "components": components,
    }
