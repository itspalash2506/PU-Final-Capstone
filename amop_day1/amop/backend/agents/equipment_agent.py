"""
Equipment Agent — runs as first node after router in the agent graph.

Responsibility:
  Extract machine identifiers from the user query using regex.
  Never use an LLM call for this — it is a pattern matching task.

ID formats handled:
  Alpha format: A1-A99, B1-B99 etc. (letter + 1-3 digits)
    -> used by Retrieval Agent for Qdrant work_orders payload filter
  Numeric format: plain integers 1-100 following "machine" keyword
    -> used by Prediction Agent for PdM telemetry lookup

Derivation rule:
  If alpha found (e.g. A6) but no explicit numeric found:
    derive numeric = int(digits stripped from alpha) = 6
    set id_assumption = True
"""

import re
import time
from typing import Optional

from backend.agents.state import AgentState
from backend.core.logging import get_logger

logger = get_logger(__name__)

_ALPHA_PATTERN = re.compile(r'\b([A-Z]\d{1,3})\b')
_NUMERIC_PATTERN = re.compile(r'\bmachine\s*#?(\d{1,3})\b', re.IGNORECASE)


def equipment_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    query = state.get("query", "")

    try:
        alpha_match = _ALPHA_PATTERN.search(query)
        numeric_match = _NUMERIC_PATTERN.search(query)

        alpha_id: Optional[str] = alpha_match.group(1) if alpha_match else None
        numeric_id: Optional[int] = None
        id_assumption = False

        if numeric_match:
            numeric_id = int(numeric_match.group(1))

        if alpha_id and numeric_id is None:
            digits = re.sub(r'[^0-9]', '', alpha_id)
            if digits:
                numeric_id = int(digits)
                id_assumption = True

        status = "success" if (alpha_id or numeric_id) else "no_id_found"
        extracted_summary = (
            f"alpha={alpha_id}, numeric={numeric_id}, assumption={id_assumption}"
        )

        latency_ms = int((time.perf_counter() - t0) * 1000)

        trace = {
            "agent_name": "equipment",
            "status": status,
            "latency_ms": latency_ms,
            "extracted": extracted_summary,
        }

        logger.info(
            "Equipment Agent complete",
            extra={"status": status, "extracted": extracted_summary},
        )

        existing_traces = list(state.get("agent_traces") or [])
        existing_traces.append(trace)

        return {
            "alpha_machine_id": alpha_id,
            "numeric_machine_id": numeric_id,
            "id_assumption": id_assumption,
            "machine_context": {
                "alpha_id": alpha_id,
                "numeric_id": numeric_id,
                "assumption": id_assumption,
                "raw_query": query,
            },
            "agent_traces": existing_traces,
        }

    except Exception as e:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning(
            "Equipment Agent failed",
            extra={"error": str(e)},
        )
        existing_traces = list(state.get("agent_traces") or [])
        existing_traces.append({
            "agent_name": "equipment",
            "status": "failed",
            "latency_ms": latency_ms,
            "extracted": f"error: {str(e)}",
        })
        return {
            "alpha_machine_id": None,
            "numeric_machine_id": None,
            "id_assumption": False,
            "machine_context": {},
            "agent_traces": existing_traces,
        }
