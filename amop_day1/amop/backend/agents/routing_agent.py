"""
Routing Agent — runs after Prediction Agent (on rag/both paths), before Summarizer.

Responsibility:
  Recommend which technician should handle this work order.
  Determine urgency level based on prediction risk tier.
  Always degrade gracefully — never crash the pipeline.

Inputs read from state:
  alpha_machine_id  -> SQLite technician history query
  prediction_result -> urgency determination
  retrieved_docs    -> fallback technician source (from rag_result sources)

Never raises an exception out of this function.
"""

import time
from collections import Counter
from typing import Optional

from backend.agents.state import AgentState
from backend.core.logging import get_logger

logger = get_logger(__name__)

_URGENCY_MAP = {
    "Critical": ("immediate", "Respond within 1 hour"),
    "High":     ("urgent",    "Respond within 4 hours"),
    "Medium":   ("scheduled", "Next maintenance window"),
    "Low":      ("monitor",   "Flag for next inspection"),
}


def _get_urgency(prediction_result: Optional[dict]) -> tuple[str, str]:
    if not prediction_result:
        return "scheduled", "No prediction available"
    risk_tier = prediction_result.get("risk_tier") or prediction_result.get("severity", "")
    # Normalize — prediction_result may come from work-order predictor (severity field)
    tier_key = risk_tier.capitalize() if risk_tier else ""
    return _URGENCY_MAP.get(tier_key, ("scheduled", "No prediction available"))


def _query_technician_history(machine_id: str) -> list[tuple[str, int]]:
    """Return [(technician, count)] ordered by count desc. Returns [] on any error."""
    try:
        from backend.db.session import SessionLocal
        from backend.db.models import WorkOrder
        from sqlalchemy import func

        db = SessionLocal()
        try:
            rows = (
                db.query(WorkOrder.primary_tech, func.count(WorkOrder.primary_tech).label("cnt"))
                .filter(WorkOrder.machine_id == machine_id)
                .filter(WorkOrder.primary_tech.isnot(None))
                .group_by(WorkOrder.primary_tech)
                .order_by(func.count(WorkOrder.primary_tech).desc())
                .limit(5)
                .all()
            )
            return [(row.primary_tech, row.cnt) for row in rows]
        finally:
            db.close()
    except Exception as exc:
        logger.warning("Routing Agent DB query failed", extra={"error": str(exc)})
        return []


def routing_node(state: AgentState) -> dict:
    t0 = time.perf_counter()

    try:
        alpha_machine_id = state.get("alpha_machine_id")
        prediction_result = state.get("prediction_result")
        rag_result = state.get("rag_result")

        urgency, urgency_detail = _get_urgency(prediction_result)

        recommended = "unassigned"
        backup = None
        basis = "unassigned"
        history_count = 0

        if alpha_machine_id:
            history = _query_technician_history(alpha_machine_id)
            history_count = sum(cnt for _, cnt in history)
            if history:
                recommended = history[0][0]
                backup = history[1][0] if len(history) >= 2 else None
                basis = "machine_history"

        # Fallback: scan retrieved docs from RAG result
        if recommended == "unassigned" and rag_result:
            sources = rag_result.get("sources", [])
            techs = [
                doc.get("primary_tech")
                for doc in sources
                if doc.get("primary_tech")
            ]
            if techs:
                counter = Counter(techs)
                most_common = counter.most_common(2)
                recommended = most_common[0][0]
                backup = most_common[1][0] if len(most_common) >= 2 else None
                basis = "document_similarity"

        routing_result = {
            "recommended_technician": recommended,
            "backup_technician": backup,
            "urgency": urgency,
            "urgency_detail": urgency_detail,
            "assignment_basis": basis,
            "technician_history_count": history_count,
        }

        status = "success" if recommended != "unassigned" else "no_history"
        latency_ms = int((time.perf_counter() - t0) * 1000)

        traces = list(state.get("agent_traces") or [])
        traces.append({
            "agent_name": "routing",
            "status": status,
            "latency_ms": latency_ms,
        })

        logger.info(
            "Routing Agent complete",
            extra={
                "recommended": recommended,
                "urgency": urgency,
                "basis": basis,
                "latency_ms": latency_ms,
            },
        )

        return {"routing_result": routing_result, "agent_traces": traces}

    except Exception as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning("Routing Agent failed", extra={"error": str(exc)})
        traces = list(state.get("agent_traces") or [])
        traces.append({
            "agent_name": "routing",
            "status": "failed",
            "latency_ms": latency_ms,
        })
        return {"routing_result": None, "agent_traces": traces}
