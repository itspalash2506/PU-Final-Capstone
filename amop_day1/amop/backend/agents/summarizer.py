"""
Summarizer Agent — consolidates per-agent results into the final answer.

For rag-only queries the RAG answer is passed through unchanged (no extra LLM call).
For both-intent queries the two sections are concatenated with clear headers.
For rca-intent queries the RCA analysis is passed through with a header.
"""

import time

from backend.agents.state import AgentState
from backend.core.logging import get_logger

logger = get_logger(__name__)


def summarizer_node(state: AgentState) -> dict:
    t0 = time.perf_counter()

    intent = state.get("intent", "rag")
    rag_result = state.get("rag_result")
    prediction_result = state.get("prediction_result")
    rca_result = state.get("rca_result")

    sources: list = []
    rag_latency = 0

    if intent == "rag":
        final_answer = rag_result["answer"] if rag_result else "No results available."
        sources = rag_result.get("sources", []) if rag_result else []
        rag_latency = rag_result.get("latency_ms", 0) if rag_result else 0

    elif intent == "predict":
        final_answer = (
            prediction_result.get("message", "Prediction unavailable.")
            if prediction_result
            else "Prediction unavailable."
        )

    elif intent == "both":
        parts: list[str] = []
        if rag_result:
            parts.append("**Historical Analysis:**\n" + rag_result["answer"])
            sources = rag_result.get("sources", [])
            rag_latency = rag_result.get("latency_ms", 0)
        if prediction_result:
            parts.append(
                "**Predictive Analysis:**\n"
                + prediction_result.get("message", "Prediction unavailable.")
            )
        final_answer = "\n\n".join(parts) if parts else "No results available."

    elif intent == "rca":
        if rca_result:
            analysis = rca_result.get("analysis", "Root cause analysis unavailable.")
            n_records = rca_result.get("num_records_analyzed", 0)
            header = f"**Root Cause Analysis** *(based on {n_records} historical record{'s' if n_records != 1 else ''})*\n\n"
            final_answer = header + analysis
            sources = rca_result.get("sources", [])
            rag_latency = rca_result.get("latency_ms", 0)
        else:
            final_answer = "Root cause analysis unavailable."

    else:
        final_answer = rag_result["answer"] if rag_result else "No results available."
        sources = rag_result.get("sources", []) if rag_result else []
        rag_latency = rag_result.get("latency_ms", 0) if rag_result else 0

    # Append routing recommendation if available (rag and both intents)
    routing = state.get("routing_result")
    if routing and routing.get("recommended_technician") != "unassigned":
        routing_section = (
            f"\n\n**Recommended Technician:** {routing['recommended_technician']}"
            f" — {routing['urgency'].upper()}: {routing['urgency_detail']}"
        )
        if routing.get("backup_technician"):
            routing_section += f"\n**Backup:** {routing['backup_technician']}"
        final_answer += routing_section

    summarizer_latency = int((time.perf_counter() - t0) * 1000)
    traces = list(state.get("agent_traces", []))
    traces.append({
        "agent_name": "summarizer",
        "status": "success",
        "latency_ms": summarizer_latency,
        "error": None,
    })

    total_latency = rag_latency + summarizer_latency

    logger.info(
        "Summarizer complete",
        extra={"intent": intent, "answer_len": len(final_answer), "sources": len(sources)},
    )

    return {
        "final_answer": final_answer,
        "sources": sources,
        "latency_ms": total_latency,
        "agent_traces": traces,
    }
