"""
Router Agent — classifies query intent using an LLM call.

Intents:
    rag     — historical record lookup / diagnosis
    predict — failure probability, severity, ETA, risk estimation
    rca     — root cause analysis: WHY did it fail, underlying cause pattern
    both    — needs historical context AND a prediction

Falls back to keyword-based routing if the LLM call fails (rate limit,
network error, etc.) so the pipeline stays functional on degraded infrastructure.
"""

import time
import re

from openai import RateLimitError, BadRequestError, NotFoundError

from backend.agents.state import AgentState
from backend.clients.openrouter import call_with_key_rotation
from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

# ── Keyword fallback sets ──────────────────────────────────────────────────────

_PREDICT_KEYWORDS = {
    "predict", "prediction", "probability", "prognosis", "forecast",
    "failure risk", "risk", "severity", "how severe", "how bad",
    "eta", "how long", "repair time", "time to repair",
    "will fail", "likely to fail", "when will", "when does",
    "estimate", "expected", "remaining life",
}

_HISTORY_KEYWORDS = {
    "history", "historical", "past", "previous", "before",
    "what happened", "what issues", "what was", "records", "work order",
    "has had", "have had", "technician", "notes",
}

_RCA_KEYWORDS = {
    "root cause", "root-cause", "rca", "why did", "why does", "why is",
    "cause of", "caused by", "reason for", "reason why",
    "underlying", "failure mode", "failure pattern",
    "what caused", "what is causing", "contributing factor",
    "failure analysis", "analyze failure", "diagnose why",
    "systematic", "recurrence", "recurring",
}

# ── LLM classification prompt ─────────────────────────────────────────────────

_ROUTER_SYSTEM = """\
You are an intent classifier for an industrial maintenance AI platform.
Classify the technician's query into EXACTLY ONE of these intents:

rag     - Looking up historical maintenance records, past work orders, what happened,
          technician notes, or diagnosing an issue using historical data.

predict - Predicting failure probability, risk level, severity, or estimated repair time
          for a machine based on its current state.

rca     - Root Cause Analysis: asking WHY something failed, what the underlying cause is,
          analyzing recurring failure patterns to find the root cause.
          Keywords: "why", "root cause", "caused by", "failure mode", "recurring", "RCA".

both    - The query needs BOTH historical record lookup AND a predictive assessment.

Respond with ONLY the single intent word (rag, predict, rca, or both). No explanation.\
"""

_ROUTER_USER = "Query: {query}"

_VALID_INTENTS = {"rag", "predict", "rca", "both"}


def _keyword_route(query: str) -> str | None:
    """
    Classify by keyword matching. Returns None when no keywords match so the
    caller knows the query is ambiguous and should fall through to the LLM.
    """
    q = query.lower()
    has_predict = any(kw in q for kw in _PREDICT_KEYWORDS)
    has_history = any(kw in q for kw in _HISTORY_KEYWORDS)
    has_rca = any(kw in q for kw in _RCA_KEYWORDS)

    if has_rca:
        return "rca"
    if has_predict and has_history:
        return "both"
    if has_predict:
        return "predict"
    if has_history:
        return "rag"
    return None  # ambiguous — no keywords matched


def _llm_route(query: str) -> tuple[str, str]:
    """LLM-based intent classification. Raises on failure."""
    settings = get_settings()

    models_to_try = [settings.openrouter_router_model] + [
        m.strip()
        for m in settings.openrouter_router_fallback_models.split(",")
        if m.strip()
    ]

    messages = [
        {"role": "system", "content": _ROUTER_SYSTEM},
        {"role": "user", "content": _ROUTER_USER.format(query=query)},
    ]

    for model in models_to_try:
        try:
            raw = call_with_key_rotation(model, messages, temperature=0.0, max_tokens=10)
            raw = raw.strip().lower()
            intent = re.sub(r"[^a-z]", "", raw.split()[0]) if raw.split() else ""
            if intent in _VALID_INTENTS:
                logger.info("LLM router classified", extra={"model": model, "intent": intent})
                return intent, model
        except (RateLimitError, BadRequestError, NotFoundError):
            logger.warning("Router LLM unavailable, trying fallback", extra={"model": model})
            continue
        except Exception as exc:
            logger.warning("Router LLM error", extra={"model": model, "error": str(exc)})
            continue

    raise RuntimeError("All LLM models exhausted for routing")


def router_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    query = state["query"]

    intent = _keyword_route(query)
    if intent is not None:
        method = "keyword"
        router_model = None
    else:
        try:
            intent, router_model = _llm_route(query)
            method = "llm"
        except Exception:
            intent = "rag"  # safe default when both keyword and LLM fail
            method = "keyword_fallback"
            router_model = None

    latency_ms = int((time.perf_counter() - t0) * 1000)
    traces = list(state.get("agent_traces", []))
    traces.append({
        "agent_name": "router",
        "status": "success",
        "latency_ms": latency_ms,
        "error": None,
        "method": method,
        "model": router_model,
    })

    logger.info(
        "Router classified query",
        extra={"intent": intent, "method": method, "latency_ms": latency_ms},
    )
    return {
        "intent": intent,
        "agent_traces": traces,
        "router_method": method,
        "router_model": router_model,
    }
