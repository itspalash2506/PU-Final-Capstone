"""
API route registry.
Each router is registered here and included in main.py via create_app().
"""

from backend.api.routes.health import router as health_router
from backend.api.routes.ingest import router as ingest_router
from backend.api.routes.query import router as query_router

__all__ = ["health_router", "ingest_router", "query_router"]
