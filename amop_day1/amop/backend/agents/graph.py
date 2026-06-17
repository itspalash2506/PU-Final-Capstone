"""
LangGraph agent graph for AMOP.

Flow:
    START
    -> equipment_agent   (always — regex ID extraction, no LLM)
    -> router            (small dedicated router model, falls back to keywords)
    -> [conditional on intent]:
        "rag"     -> rag -> routing -> summarizer -> END
        "predict" -> prediction -> summarizer -> END
        "both"    -> rag -> rca -> prediction -> routing -> summarizer -> END
        "rca"     -> rag -> rca -> summarizer -> END
"""

import asyncio
import threading
import time
from typing import AsyncGenerator, Optional

from langgraph.graph import StateGraph, END, START

from backend.agents.state import AgentState
from backend.agents.equipment_agent import equipment_node
from backend.agents.router import router_node
from backend.agents.rag_agent import rag_node
from backend.agents.prediction_agent import prediction_node
from backend.agents.rca_agent import rca_node
from backend.agents.routing_agent import routing_node
from backend.agents.summarizer import summarizer_node
from backend.agents.token_stream import register as _ts_register, unregister as _ts_unregister
from backend.core.logging import get_logger

logger = get_logger(__name__)


def _route_after_router(state: AgentState) -> str:
    intent = state.get("intent", "rag")
    return intent if intent in ("rag", "predict", "both", "rca") else "rag"


def _route_after_rag(state: AgentState) -> str:
    """After RAG: only invoke RCA when the user explicitly wants root-cause
    analysis ('rca') or asked for both history+prediction ('both').
    Plain 'rag' history lookups skip RCA — it added ~75s of extra latency
    (a redundant run_rag + second LLM call) with no benefit for simple queries.
    """
    intent = state.get("intent", "rag")
    if intent in ("rca", "both"):
        return "rca"
    return "routing"


def _route_after_rca(state: AgentState) -> str:
    """After RCA: predict-and-route for 'both', route-only for 'rag', summarize for 'rca'."""
    intent = state.get("intent", "rag")
    if intent == "both":
        return "prediction"
    if intent == "rca":
        return "summarizer"
    # "rag" intent -> routing then summarizer
    return "routing"


def _route_after_prediction(state: AgentState) -> str:
    """After prediction: run routing for 'both', summarize for 'predict'."""
    intent = state.get("intent", "predict")
    if intent == "both":
        return "routing"
    return "summarizer"


def _build_graph():
    g: StateGraph = StateGraph(AgentState)

    g.add_node("equipment", equipment_node)
    g.add_node("router", router_node)
    g.add_node("rag", rag_node)
    g.add_node("prediction", prediction_node)
    g.add_node("rca", rca_node)
    g.add_node("routing", routing_node)
    g.add_node("summarizer", summarizer_node)

    # Equipment agent always runs first (no LLM, sub-millisecond)
    g.add_edge(START, "equipment")
    g.add_edge("equipment", "router")

    # Router fans out to rag, prediction, or rca-via-rag
    g.add_conditional_edges(
        "router",
        _route_after_router,
        {"rag": "rag", "predict": "prediction", "both": "rag", "rca": "rag"},
    )

    # RAG routes: rca/both intents go through RCA; plain rag skips it
    g.add_conditional_edges(
        "rag",
        _route_after_rag,
        {"rca": "rca", "routing": "routing"},
    )

    # RCA routes: both -> prediction, rca -> summarizer, rag -> routing
    g.add_conditional_edges(
        "rca",
        _route_after_rca,
        {"prediction": "prediction", "routing": "routing", "summarizer": "summarizer"},
    )

    # Prediction routes: both -> routing, predict -> summarizer
    g.add_conditional_edges(
        "prediction",
        _route_after_prediction,
        {"routing": "routing", "summarizer": "summarizer"},
    )

    g.add_edge("routing", "summarizer")
    g.add_edge("summarizer", END)

    return g.compile()


# Graph is compiled once at import time
_graph = _build_graph()


def run_agent_graph(
    query: str,
    top_k: int = 5,
    machine_id: Optional[str] = None,
) -> dict:
    """
    Run the full multi-agent graph and return a result dict:
        answer              str
        sources             list[dict]
        latency_ms          int
        intent              str
        agent_traces        list[dict]
        routing_result      dict | None
        rca_result          dict | None
        alpha_machine_id    str | None
        numeric_machine_id  int | None
        id_assumption       bool
    """
    t0 = time.perf_counter()

    initial_state: AgentState = {
        "query": query,
        "machine_id": machine_id,
        "top_k": top_k,
        "intent": "rag",
        "router_method": None,
        "router_model": None,
        # Equipment fields (populated by equipment_node)
        "alpha_machine_id": None,
        "numeric_machine_id": None,
        "id_assumption": False,
        "machine_context": {},
        # Agent results
        "rag_result": None,
        "prediction_result": None,
        "rca_result": None,
        "routing_result": None,
        # Output
        "final_answer": "",
        "sources": [],
        "latency_ms": 0,
        "agent_traces": [],
    }

    final_state = _graph.invoke(initial_state)

    wall_latency = int((time.perf_counter() - t0) * 1000)

    logger.info(
        "Agent graph complete",
        extra={
            "intent": final_state.get("intent"),
            "wall_latency_ms": wall_latency,
            "num_traces": len(final_state.get("agent_traces", [])),
        },
    )

    return {
        "answer": final_state.get("final_answer", ""),
        "sources": final_state.get("sources", []),
        "latency_ms": wall_latency,
        "intent": final_state.get("intent", "rag"),
        "agent_traces": final_state.get("agent_traces", []),
        "routing_result": final_state.get("routing_result"),
        "rca_result": final_state.get("rca_result"),
        "router_method": final_state.get("router_method"),
        "router_model": final_state.get("router_model"),
        "alpha_machine_id": final_state.get("alpha_machine_id"),
        "numeric_machine_id": final_state.get("numeric_machine_id"),
        "id_assumption": final_state.get("id_assumption", False),
    }


# ── Human-readable labels for each agent node ─────────────────────────────────

_AGENT_LABELS = {
    "equipment": "Equipment Agent",
    "router": "Intent Router",
    "rag": "RAG Agent",
    "rca": "RCA Agent",
    "prediction": "Prediction Agent",
    "routing": "Routing Agent",
    "summarizer": "Summarizer",
}


def _build_agent_event(node_name: str, updates: dict, accumulated: dict) -> dict:
    """Build a structured SSE event dict for a completed agent node."""
    traces = accumulated.get("agent_traces", [])
    node_trace = next(
        (t for t in reversed(traces) if t.get("agent_name") == node_name), {}
    )

    event: dict = {
        "type": "agent_complete",
        "agent": node_name,
        "label": _AGENT_LABELS.get(node_name, node_name.title()),
        "status": node_trace.get("status", "success"),
        "latency_ms": node_trace.get("latency_ms"),
        "summary": "",
        "data": {},
    }

    if node_name == "equipment":
        alpha = updates.get("alpha_machine_id")
        numeric = updates.get("numeric_machine_id")
        assumption = updates.get("id_assumption", False)
        if alpha:
            suffix = f" (#{numeric}, inferred)" if assumption else f" (#{numeric})" if numeric else ""
            event["summary"] = f"Machine ID: {alpha}{suffix}"
        else:
            event["summary"] = "No machine ID found in query"
        event["data"] = {
            "alpha_machine_id": alpha,
            "numeric_machine_id": numeric,
            "id_assumption": assumption,
        }

    elif node_name == "router":
        intent = updates.get("intent") or accumulated.get("intent", "rag")
        method = updates.get("router_method") or accumulated.get("router_method")
        model = updates.get("router_model") or accumulated.get("router_model")
        intent_labels = {
            "rag": "History Lookup",
            "predict": "Failure Prediction",
            "both": "History + Prediction",
            "rca": "Root Cause Analysis",
        }
        if method == "llm" and model:
            event["summary"] = f"Intent: {intent_labels.get(intent, intent.upper())} via fast router model"
        elif method:
            event["summary"] = f"Intent: {intent_labels.get(intent, intent.upper())} via {method.replace('_', ' ')}"
        else:
            event["summary"] = f"Intent: {intent_labels.get(intent, intent.upper())}"
        event["data"] = {"intent": intent, "method": method, "model": model}

    elif node_name == "rag":
        rag_result = updates.get("rag_result") or {}
        sources = rag_result.get("sources", [])
        event["summary"] = f"Retrieved {len(sources)} relevant work orders"
        event["data"] = {"source_count": len(sources)}

    elif node_name == "rca":
        rca_result = updates.get("rca_result") or {}
        cause = rca_result.get("probable_cause") or rca_result.get("analysis", "")
        confidence = rca_result.get("confidence", "")
        short = str(cause)[:100] + ("..." if len(str(cause)) > 100 else "")
        event["summary"] = short or "Analysis complete"
        event["data"] = {"confidence": confidence}

    elif node_name == "prediction":
        pred = updates.get("prediction_result") or {}
        severity = pred.get("severity", "unknown")
        eta = pred.get("eta_hours")
        event["summary"] = f"Severity: {severity.upper()}" + (f", ETA: {eta}h" if eta else "")
        event["data"] = {k: v for k, v in pred.items() if k in ("severity", "eta_hours", "severity_confidence")}

    elif node_name == "routing":
        routing = updates.get("routing_result") or {}
        tech = routing.get("recommended_technician", "TBD")
        urgency = routing.get("urgency", "unknown")
        event["summary"] = f"Assigned: {tech} — {urgency} priority"
        event["data"] = routing

    elif node_name == "summarizer":
        event["summary"] = "Final answer generated"

    return event


async def stream_agent_graph(
    query: str,
    top_k: int = 5,
    machine_id: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """
    Async generator that yields SSE event dicts as each agent node completes,
    followed by a final 'complete' event with the full result.

    Runs the synchronous LangGraph .stream() in a background thread and bridges
    results into the async event loop via asyncio.Queue.
    """
    initial_state: AgentState = {
        "query": query,
        "machine_id": machine_id,
        "top_k": top_k,
        "intent": "rag",
        "router_method": None,
        "router_model": None,
        "alpha_machine_id": None,
        "numeric_machine_id": None,
        "id_assumption": False,
        "machine_context": {},
        "rag_result": None,
        "prediction_result": None,
        "rca_result": None,
        "routing_result": None,
        "final_answer": "",
        "sources": [],
        "latency_ms": 0,
        "agent_traces": [],
    }

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    t0 = time.perf_counter()

    def _run_stream():
        thread_id = threading.current_thread().ident
        _ts_register(thread_id, loop, q)
        try:
            accumulated: dict = dict(initial_state)
            for chunk in _graph.stream(initial_state, stream_mode="updates"):
                node_name = next(iter(chunk))
                updates = chunk[node_name]
                accumulated.update(updates)
                event = _build_agent_event(node_name, updates, accumulated)
                loop.call_soon_threadsafe(q.put_nowait, ("event", event, dict(accumulated)))
        except Exception as exc:
            loop.call_soon_threadsafe(q.put_nowait, ("error", str(exc), {}))
        finally:
            _ts_unregister(thread_id)
            loop.call_soon_threadsafe(q.put_nowait, ("done", None, {}))

    thread = threading.Thread(target=_run_stream, daemon=True)
    thread.start()

    final_accumulated: dict = {}

    while True:
        try:
            msg_type, payload, state_snap = await asyncio.wait_for(q.get(), timeout=120.0)
        except asyncio.TimeoutError:
            yield {"type": "error", "message": "Pipeline timed out after 120 seconds"}
            return

        if msg_type == "error":
            yield {"type": "error", "message": payload}
            return

        if msg_type == "done":
            wall_ms = int((time.perf_counter() - t0) * 1000)
            yield {
                "type": "complete",
                "answer": final_accumulated.get("final_answer", ""),
                "sources": final_accumulated.get("sources", []),
                "intent": final_accumulated.get("intent", "rag"),
                "agent_traces": final_accumulated.get("agent_traces", []),
                "routing_result": final_accumulated.get("routing_result"),
                "rca_result": final_accumulated.get("rca_result"),
                "router_method": final_accumulated.get("router_method"),
                "router_model": final_accumulated.get("router_model"),
                "alpha_machine_id": final_accumulated.get("alpha_machine_id"),
                "numeric_machine_id": final_accumulated.get("numeric_machine_id"),
                "id_assumption": final_accumulated.get("id_assumption", False),
                "latency_ms": wall_ms,
            }
            return

        # msg_type == "event"
        # Token events carry an empty state_snap — don't overwrite accumulated state with {}.
        if state_snap:
            final_accumulated = state_snap
        yield payload
