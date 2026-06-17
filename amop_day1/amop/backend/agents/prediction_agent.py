"""
Prediction Agent — calls the XGBoost failure severity + ETA predictor.

Falls back gracefully if the model has not been trained yet (before first ingest).
"""

import time

from backend.agents.state import AgentState
from backend.prediction.predictor import predict, ModelNotTrainedError
from backend.core.logging import get_logger

logger = get_logger(__name__)


def prediction_node(state: AgentState) -> dict:
    t0 = time.perf_counter()

    machine_id = state.get("machine_id") or ""
    issue = state.get("query", "")

    try:
        result = predict(
            machine_id=machine_id,
            issue_description=issue,
            technician_notes=None,
        )
        result["available"] = True
        result["message"] = (
            f"Severity: {result['severity'].upper()} "
            f"(confidence {result['severity_confidence']:.0%}), "
            f"estimated repair time: {result['eta_hours']} hours."
        )
        status = "success"
        error = None

    except ModelNotTrainedError:
        result = {
            "severity": "unknown",
            "severity_confidence": 0.0,
            "eta_hours": 0.0,
            "features_used": [],
            "available": False,
            "message": (
                "Predictive analysis is not yet available — "
                "the model will be trained automatically after the first data ingest."
            ),
        }
        status = "skipped"
        error = None

    except Exception as exc:
        logger.error("Prediction agent failed", extra={"error": str(exc)}, exc_info=True)
        result = {
            "severity": "unknown",
            "severity_confidence": 0.0,
            "eta_hours": 0.0,
            "features_used": [],
            "available": False,
            "message": f"Prediction failed: {exc}",
        }
        status = "failed"
        error = str(exc)

    latency_ms = int((time.perf_counter() - t0) * 1000)
    traces = list(state.get("agent_traces", []))
    traces.append({
        "agent_name": "prediction",
        "status": status,
        "latency_ms": latency_ms,
        "error": error,
    })

    return {"prediction_result": result, "agent_traces": traces}
