import uuid
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core.config import get_settings
from backend.core.logging import setup_logging, get_logger
from backend.db.session import init_db
from backend.api.routes.health import router as health_router
from backend.api.routes.ingest import router as ingest_router
from backend.api.routes.query import router as query_router
from backend.api.routes.predict import router as predict_router
from backend.api.routes.pdm import router as pdm_router
from backend.api.routes.eval import router as eval_router
from backend.api.routes.feedback import router as feedback_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown logic.
    FastAPI calls this once on start (before serving) and on shutdown.
    """
    settings = get_settings()

    # ── Startup ───────────────────────────────────────────────────────────────
    setup_logging(settings.log_level)
    logger.info("AMOP starting up", extra={"env": settings.app_env})

    # Initialize SQLite schema (idempotent)
    init_db()

    # Build PdM machine profiles in Qdrant if not already built.
    # Non-blocking: a Qdrant connection failure here does not prevent the API
    # from serving work-order queries.
    try:
        from backend.prediction.pdm_semantic_bridge import build_profiles_if_needed
        built = build_profiles_if_needed()
        if built:
            logger.info("PdM machine profiles built and stored in Qdrant")
        else:
            logger.info("PdM machine profiles already present in Qdrant")
    except Exception as e:
        logger.warning("PdM profile build failed (non-fatal)", extra={"error": str(e)})

    # Eagerly load the cross-encoder reranker so the first real request pays
    # no extra latency and startup failure surfaces here rather than mid-request.
    try:
        from backend.rag.reranker import get_reranker
        get_reranker()
        logger.info("Cross-encoder reranker model loaded")
    except Exception as e:
        logger.warning("Cross-encoder reranker failed to load (non-fatal)", extra={"error": str(e)})

    logger.info("AMOP startup complete")

    yield  # Application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("AMOP shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="AI Maintenance Operations Platform",
        description=(
            "Maintenance knowledge retrieval, root cause analysis, "
            "failure risk prediction, and work order routing."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Allow React dev server (port 3000) in development
    origins = ["http://localhost:3000", "http://localhost:80", "http://frontend"]
    if settings.app_env == "development":
        origins.append("*")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request ID middleware ─────────────────────────────────────────────────
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()

        response = await call_next(request)

        latency_ms = int((time.perf_counter() - start) * 1000)
        response.headers["X-Request-ID"] = request_id

        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            },
        )
        return response

    # ── Global exception handler ───────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception",
            extra={
                "request_id": getattr(request.state, "request_id", "unknown"),
                "path": request.url.path,
                "error": str(exc),
            },
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(health_router, prefix="/api/v1", tags=["health"])
    app.include_router(feedback_router, prefix="/api/v1", tags=["feedback"])
    app.include_router(ingest_router, prefix="/api/v1", tags=["ingest"])
    app.include_router(query_router, prefix="/api/v1", tags=["query"])
    app.include_router(predict_router, prefix="/api/v1", tags=["prediction"])
    app.include_router(pdm_router, prefix="/api/v1", tags=["pdm"])
    app.include_router(eval_router, prefix="/api/v1", tags=["evaluation"])

    @app.get("/", include_in_schema=False)
    async def root():
        return {"service": "AMOP", "status": "running", "docs": "/docs"}

    return app


# Entry point for uvicorn:  uvicorn backend.api.main:app
app = create_app()
