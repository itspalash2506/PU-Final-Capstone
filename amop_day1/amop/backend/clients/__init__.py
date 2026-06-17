"""
External service clients.
"""

from backend.clients.embedding_client import encode
from backend.clients.vector_store import (
    get_qdrant_client,
    ensure_collection,
    upsert_points,
    search,
)
from backend.clients.openrouter import get_openrouter_client

__all__ = [
    "encode",
    "get_qdrant_client",
    "ensure_collection",
    "upsert_points",
    "search",
    "get_openrouter_client",
]
