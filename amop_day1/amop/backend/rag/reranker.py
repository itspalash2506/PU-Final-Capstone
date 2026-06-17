"""
Cross-encoder reranker using sentence-transformers.

Why cross-encoder over LLM reranking:
- Cross-encoders read the query AND document together as a pair, giving much
  better relevance scoring than embedding similarity alone.
- Zero API cost, runs locally, adds ~100-150ms latency on CPU.
- ms-marco-MiniLM-L-6-v2 is trained specifically for passage reranking.

When to use:
- Only on RAG or BOTH intent paths (not prediction-only).
- Applied after RRF fusion, before returning results to the agent.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
"""

import math
import time
from datetime import datetime, timezone
from functools import lru_cache

from sentence_transformers import CrossEncoder

from backend.core.logging import get_logger

logger = get_logger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ── Model loader ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load and cache the cross-encoder model (downloads ~80MB on first call)."""
    logger.info("Loading cross-encoder model", extra={"model": _MODEL_NAME})
    model = CrossEncoder(_MODEL_NAME)
    logger.info("Cross-encoder model ready", extra={"model": _MODEL_NAME})
    return model


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid to map raw CE logit → [0, 1]."""
    return 1.0 / (1.0 + math.exp(-x))


def _recency_scores(docs: list[dict]) -> list[float]:
    """
    Normalize created_at timestamps within the result set to [0, 1].
    Most recent doc → 1.0, oldest → 0.0, linear in between.
    Falls back to 0.5 for missing/unparseable dates or if all dates are equal.
    """
    parsed: list[datetime | None] = []
    for doc in docs:
        raw = doc.get("created_at")
        if raw:
            try:
                dt = datetime.fromisoformat(raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                parsed.append(dt)
            except (ValueError, TypeError):
                parsed.append(None)
        else:
            parsed.append(None)

    valid = [d for d in parsed if d is not None]
    if len(valid) < 2:
        return [0.5] * len(docs)

    min_dt, max_dt = min(valid), max(valid)
    span = (max_dt - min_dt).total_seconds()
    if span == 0:
        return [0.5] * len(docs)

    return [
        round((dt - min_dt).total_seconds() / span, 6) if dt is not None else 0.5
        for dt in parsed
    ]


# ── Public interface ──────────────────────────────────────────────────────────

def rerank(query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank docs using a cross-encoder and return top_k by composite score.

    Composite score:
        final = 0.7 * sigmoid(raw_ce_score)
              + 0.2 * recency_score          (0.5 default until real dates)
              + 0.1 * effectiveness_score    (payload field or 0.5 default)

    Each returned doc gets a "rerank_score" key added.
    Falls back to original RRF order if the cross-encoder fails.
    """
    if not docs:
        return docs

    try:
        t0 = time.perf_counter()

        model = get_reranker()

        pairs = [
            (query, f"{doc['issue_description']} {doc.get('technician_notes') or ''}")
            for doc in docs
        ]

        raw_scores: list[float] = model.predict(pairs).tolist()
        recency_scores = _recency_scores(docs)

        scored = []
        for doc, raw_ce, recency in zip(docs, raw_scores, recency_scores):
            ce_norm = _sigmoid(raw_ce)
            effectiveness = float(doc.get("effectiveness_score", 0.5))

            final = 0.7 * ce_norm + 0.2 * recency + 0.1 * effectiveness

            scored.append({**doc, "score": round(final, 6)})

        scored.sort(key=lambda d: d["score"], reverse=True)
        result = scored[:top_k]

        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "Reranking complete",
            extra={"input_docs": len(docs), "returned": len(result), "latency_ms": latency_ms},
        )
        return result

    except Exception as exc:
        logger.warning(
            "Cross-encoder reranking failed — falling back to RRF order",
            extra={"error": str(exc)},
        )
        return docs[:top_k]
