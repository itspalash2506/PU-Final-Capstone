"""
Hybrid retriever: BM25 (sparse) + Qdrant dense search, fused via Reciprocal Rank Fusion.

Why hybrid?
- BM25 excels at exact keyword matches (machine IDs, part numbers, error codes).
- Dense search captures semantic meaning ("motor won't start" ≈ "motor failure on startup").
- RRF fusion gives a single ranked list that outperforms either method alone.
"""

import os
import pickle
from functools import lru_cache
from typing import Optional

from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.clients.embedding_client import encode
from backend.clients.vector_store import get_qdrant_client, search as qdrant_search

logger = get_logger(__name__)

_RRF_K = 60  # standard constant — larger values flatten score differences across ranks


# ── Index loader ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_bm25() -> Optional[dict]:
    """Load BM25 index from disk once and cache in memory."""
    path = get_settings().bm25_index_path
    if not os.path.exists(path):
        logger.warning("BM25 index not found; dense-only search will be used")
        return None
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    logger.info("BM25 index loaded", extra={"docs": len(bundle["doc_ids"]), "path": path})
    return bundle


def reload_bm25() -> None:
    """Call after re-ingestion to force the cached index to refresh."""
    _load_bm25.cache_clear()


# ── Individual retrievers ─────────────────────────────────────────────────────

def _dense_search(query: str, top_k: int, machine_id: Optional[str]) -> list[tuple[str, dict]]:
    """Return [(doc_id, payload)] from Qdrant, ordered by cosine similarity."""
    query_vector = encode([query])[0]
    hits = qdrant_search(query_vector, top_k=top_k, machine_id=machine_id)
    return [(str(p.id), p.payload or {}) for p in hits]


def _sparse_search(query: str, top_k: int) -> list[str]:
    """Return doc_ids ranked by BM25 score. Returns [] if index not available."""
    bundle = _load_bm25()
    if bundle is None:
        return []
    bm25 = bundle["index"]
    doc_ids = bundle["doc_ids"]
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [doc_ids[i] for i, s in ranked if s > 0]


# ── RRF fusion ────────────────────────────────────────────────────────────────

def _rrf_merge(dense_ids: list[str], sparse_ids: list[str]) -> list[str]:
    """
    Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank(d)).
    Ranks are 1-indexed. Returns doc IDs sorted by descending combined score.
    """
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank)
    for rank, doc_id in enumerate(sparse_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


# ── Public interface ──────────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = 5,
    machine_id: Optional[str] = None,
    rerank: bool = False,
) -> list[dict]:
    """
    Hybrid retrieval returning up to top_k result dicts.

    Each dict contains:
        id, machine_id, issue_description, technician_notes, score

    Pass rerank=True to run the cross-encoder reranker after RRF fusion.
    """
    settings = get_settings()
    fetch_k = top_k * 2  # over-fetch so fusion has more candidates

    dense_hits = _dense_search(query, fetch_k, machine_id)
    dense_ids = [doc_id for doc_id, _ in dense_hits]
    payload_map: dict[str, dict] = {doc_id: pl for doc_id, pl in dense_hits}

    sparse_ids = _sparse_search(query, fetch_k)

    merged = _rrf_merge(dense_ids, sparse_ids)[:top_k]

    # Fetch payloads for any BM25-only docs not returned by dense search
    missing = [doc_id for doc_id in merged if doc_id not in payload_map]
    if missing:
        client = get_qdrant_client()
        fetched = client.retrieve(
            collection_name=settings.qdrant_collection,
            ids=missing,
            with_payload=True,
        )
        for point in fetched:
            payload_map[str(point.id)] = point.payload or {}

    results = []
    for rank, doc_id in enumerate(merged, start=1):
        payload = payload_map.get(doc_id)
        if not payload:
            continue
        # Apply machine_id filter for BM25-only hits (Qdrant already filters dense hits)
        if machine_id and payload.get("machine_id") != machine_id:
            continue
        results.append({
            "id": doc_id,
            "machine_id": payload.get("machine_id", ""),
            "issue_description": payload.get("issue_description", ""),
            "technician_notes": payload.get("technician_notes"),
            "primary_tech": payload.get("primary_tech"),
            "created_at": payload.get("created_at"),
            "effectiveness_score": payload.get("effectiveness_score", 0.5),
            "score": round(1.0 / (_RRF_K + rank), 6),
        })

    if rerank and results:
        from backend.rag.reranker import rerank as rerank_results
        results = rerank_results(query, results, top_k=top_k)

    logger.info(
        "Hybrid retrieval complete",
        extra={"dense": len(dense_ids), "sparse": len(sparse_ids), "returned": len(results), "reranked": rerank},
    )
    return results
