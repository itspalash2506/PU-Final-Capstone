"""
Thin wrapper around qdrant_client for collection management and vector search.
"""

from typing import Optional
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    ScoredPoint,
)

from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ensure_collection() -> None:
    """Create the work_orders collection if it does not exist."""
    settings = get_settings()
    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Qdrant collection created", extra={"collection": settings.qdrant_collection})


def upsert_points(points: list[PointStruct]) -> int:
    """Upsert a list of pre-built PointStructs. Returns count upserted."""
    settings = get_settings()
    client = get_qdrant_client()
    client.upsert(collection_name=settings.qdrant_collection, points=points)
    return len(points)


def search(
    query_vector: list[float],
    top_k: int = 5,
    machine_id: Optional[str] = None,
) -> list[ScoredPoint]:
    """
    Dense vector search with an optional machine_id filter.
    Returns scored Qdrant points ordered by descending cosine similarity.
    """
    settings = get_settings()
    client = get_qdrant_client()

    query_filter = None
    if machine_id:
        query_filter = Filter(
            must=[FieldCondition(key="machine_id", match=MatchValue(value=machine_id))]
        )

    # client.search() was removed in qdrant-client 1.10+; use query_points() instead.
    result = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )
    return result.points
