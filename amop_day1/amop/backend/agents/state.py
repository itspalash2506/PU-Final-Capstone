from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    query: str
    machine_id: Optional[str]
    top_k: int

    # ── Routing ───────────────────────────────────────────────────────────────
    intent: str  # "rag" | "predict" | "both" | "rca"
    router_method: Optional[str]
    router_model: Optional[str]

    # ── Equipment extraction (set by Equipment Agent) ─────────────────────────
    alpha_machine_id: Optional[str]   # e.g. "A6" — for RAG/Qdrant filter
    numeric_machine_id: Optional[int] # e.g. 6 — for PdM prediction
    id_assumption: bool               # True if numeric derived from alpha prefix
    machine_context: dict             # Equipment Agent extraction summary

    # ── Per-agent results ─────────────────────────────────────────────────────
    rag_result: Optional[dict]         # {answer, sources, latency_ms}
    prediction_result: Optional[dict]  # {severity, eta_hours, message, available}
    rca_result: Optional[dict]         # {analysis, probable_cause, recommendations, confidence, sources}
    routing_result: Optional[dict]     # {recommended_technician, urgency, urgency_detail, ...}

    # ── Consolidated output ───────────────────────────────────────────────────
    final_answer: str
    sources: list
    latency_ms: int

    # ── Observability ─────────────────────────────────────────────────────────
    agent_traces: list  # [{agent_name, status, latency_ms, error}]
