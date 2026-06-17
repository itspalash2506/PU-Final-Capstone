"""
End-to-end RAG orchestration: retrieve → generate.
Timing covers the full round trip so latency_ms is accurate in QueryResponse.
"""

import re
import time
from typing import Optional

from backend.rag.retriever import retrieve
from backend.rag.generator import generate
from backend.core.logging import get_logger

logger = get_logger(__name__)

# Matches work-order machine IDs: one uppercase letter followed by digits (e.g. A56, M3)
_MACHINE_ID_RE = re.compile(r'\b([A-Z]\d+)\b')


def _extract_machine_id(query: str) -> Optional[str]:
    """Return the first machine ID found in the query string, or None."""
    match = _MACHINE_ID_RE.search(query)
    return match.group(1) if match else None


def run_rag(
    query: str,
    top_k: int = 5,
    machine_id: Optional[str] = None,
    rerank: bool = True,
    generate_answer: bool = True,
) -> dict:
    """
    Run the full RAG pipeline and return a result dict matching QueryResponse fields:
        answer       str
        sources      list[dict]   (id, machine_id, issue_description, technician_notes, score)
        latency_ms   int

    If machine_id is not explicitly provided, it is auto-extracted from the query
    text so that Qdrant's payload filter can pin results to the right machine.

    Pass generate_answer=False to skip the LLM generation step and return only
    retrieved docs — used by the RAG agent on rca intent so the RCA agent is the
    sole LLM caller and we avoid a wasted generation round-trip.
    """
    t0 = time.perf_counter()

    if machine_id is None:
        machine_id = _extract_machine_id(query)
        if machine_id:
            logger.info("Machine ID extracted from query", extra={"machine_id": machine_id})

    docs = retrieve(query, top_k=top_k, machine_id=machine_id, rerank=rerank)

    if generate_answer:
        answer = generate(query, docs)
    else:
        answer = ""

    latency_ms = int((time.perf_counter() - t0) * 1000)

    logger.info(
        "RAG pipeline complete",
        extra={"latency_ms": latency_ms, "docs_used": len(docs), "generated": generate_answer},
    )

    return {
        "answer": answer,
        "sources": docs,
        "latency_ms": latency_ms,
    }
