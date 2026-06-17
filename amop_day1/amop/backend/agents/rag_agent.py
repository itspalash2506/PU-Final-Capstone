"""
RAG Agent — wraps the Phase 3 hybrid retrieval + generation pipeline.
"""

import time

from backend.agents.state import AgentState
from backend.rag.pipeline import run_rag
from backend.core.logging import get_logger

logger = get_logger(__name__)


def rag_node(state: AgentState) -> dict:
    t0 = time.perf_counter()

    try:
        machine_id = state.get("alpha_machine_id") or state.get("machine_id")
        # Skip LLM generation for rca intent — the RCA agent is the sole LLM caller,
        # so generating an answer here would be a wasted round-trip (~13-15s).
        intent = state.get("intent", "rag")
        result = run_rag(
            query=state["query"],
            top_k=state.get("top_k", 5),
            machine_id=machine_id,
            rerank=True,
            generate_answer=(intent != "rca"),
        )
        status = "success"
        error = None
    except Exception as exc:
        logger.error("RAG agent failed", extra={"error": str(exc)}, exc_info=True)
        result = {"answer": f"Retrieval failed: {exc}", "sources": [], "latency_ms": 0}
        status = "failed"
        error = str(exc)

    latency_ms = int((time.perf_counter() - t0) * 1000)
    traces = list(state.get("agent_traces", []))
    traces.append({
        "agent_name": "rag",
        "status": status,
        "latency_ms": latency_ms,
        "error": error,
    })

    return {"rag_result": result, "agent_traces": traces}
