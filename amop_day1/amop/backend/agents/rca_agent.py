"""
RCA Agent — Root Cause Analysis using historical records + LLM reasoning.

Flow:
    1. Retrieve historical work orders for the machine via the RAG pipeline
    2. Feed retrieved records into a specialized RCA prompt
    3. LLM synthesizes: probable root cause, contributing factors, recommendations
    4. Returns structured RCA result with evidence citations

Follows the same fallback pattern as generator.py — tries primary model then
configured fallbacks so a single rate-limit does not break the pipeline.
"""

import time

from openai import RateLimitError, BadRequestError, NotFoundError

from backend.agents.state import AgentState
from backend.rag.pipeline import run_rag
from backend.clients.openrouter import call_with_key_rotation, stream_with_key_rotation
from backend.agents.token_stream import push_token
from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

_MAX_CTX_CHARS = 4000

_RCA_SYSTEM = """\
You are an expert failure analysis engineer performing a Root Cause Analysis (RCA)
for an industrial maintenance team.

Given a set of historical work order records and a technician's question, identify:
1. The PRIMARY root cause (the underlying systemic or mechanical reason for failures)
2. Contributing factors (conditions that enabled or worsened the failure)
3. Evidence (cite specific record numbers [N] and machine IDs that support your conclusion)
4. Corrective action (what to fix now)
5. Preventive action (how to prevent recurrence)
6. Confidence level: High (3+ consistent records), Medium (1–2 records), or Low (indirect evidence only)

Format your response EXACTLY as:
**Root Cause:** [one concise sentence]
**Contributing Factors:** [bullet points or short list]
**Evidence:** [cite record numbers and machine IDs]
**Corrective Action:** [specific immediate fix]
**Preventive Action:** [systemic change to prevent recurrence]
**Confidence:** [High / Medium / Low — and why]

Rules:
- Only use facts present in the provided records. Never fabricate part numbers or procedures.
- If records are insufficient, say so explicitly in the Root Cause section.
- Each section must be 1–3 sentences or bullet points. Do not pad or repeat information across sections.
- Stop writing as soon as each section is complete.\
"""

_RCA_USER = """\
Historical maintenance records:
{context}

Technician's root cause question: {query}

Perform the RCA:\
"""


def _build_rca_context(docs: list[dict]) -> str:
    parts: list[str] = []
    total = 0
    for i, doc in enumerate(docs, start=1):
        notes = doc.get("technician_notes") or "No technician notes recorded."
        if len(notes) > 500:
            notes = notes[:500] + "…"
        entry = (
            f"[{i}] Machine: {doc['machine_id']}\n"
            f"     Issue: {doc['issue_description']}\n"
            f"     Notes: {notes}"
        )
        if total + len(entry) > _MAX_CTX_CHARS:
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n".join(parts) if parts else "No relevant records found."


def _call_rca_llm(query: str, docs: list[dict]) -> str:
    settings = get_settings()
    context = _build_rca_context(docs)

    messages = [
        {"role": "system", "content": _RCA_SYSTEM},
        {"role": "user", "content": _RCA_USER.format(context=context, query=query)},
    ]

    models_to_try = [settings.openrouter_model] + [
        m.strip()
        for m in settings.openrouter_fallback_models.split(",")
        if m.strip()
    ]

    last_exc: Exception | None = None
    for model in models_to_try:
        try:
            content = stream_with_key_rotation(
                model,
                messages,
                token_callback=lambda tok: push_token("rca", tok),
                temperature=0.1,
            )
            logger.info("RCA LLM generation successful", extra={"model": model})
            return content
        except (RateLimitError, BadRequestError, NotFoundError, ValueError) as exc:
            logger.warning("RCA model unavailable, trying fallback", extra={"model": model, "error": str(exc)})
            last_exc = exc
            continue
        except Exception as exc:
            logger.error("RCA LLM failed", extra={"model": model, "error": str(exc)}, exc_info=True)
            last_exc = exc
            break

    settings_obj = get_settings()
    if settings_obj.app_env == "development":
        return f"[RCA LLM ERROR — {type(last_exc).__name__}]: {last_exc}"
    return (
        "Root cause analysis could not be completed due to a service error. "
        "Please check connectivity and try again."
    )


def rca_node(state: AgentState) -> dict:
    t0 = time.perf_counter()

    query = state["query"]
    machine_id = state.get("machine_id")
    top_k = state.get("top_k", 8)

    status = "success"
    error = None
    sources: list = []

    try:
        # Reuse docs already retrieved by rag_node — avoids a duplicate
        # run_rag() call (full retrieval + LLM generation, ~15-30s extra).
        existing_rag = state.get("rag_result") or {}
        docs = existing_rag.get("sources", [])
        rag_latency_ms = existing_rag.get("latency_ms", 0)
        if not docs:
            # Fallback: rag_node didn't run (standalone rca intent path)
            rag_result = run_rag(
                query=query,
                top_k=max(top_k, 8),
                machine_id=machine_id,
            )
            docs = rag_result.get("sources", [])
            rag_latency_ms = rag_result.get("latency_ms", 0)
        sources = docs

        analysis_text = _call_rca_llm(query, docs)

        rca_result = {
            "analysis": analysis_text,
            "sources": docs,
            "latency_ms": rag_latency_ms,
            "num_records_analyzed": len(docs),
        }

    except Exception as exc:
        logger.error("RCA agent failed", extra={"error": str(exc)}, exc_info=True)
        rca_result = {
            "analysis": f"Root cause analysis failed: {exc}",
            "sources": [],
            "latency_ms": 0,
            "num_records_analyzed": 0,
        }
        status = "failed"
        error = str(exc)

    latency_ms = int((time.perf_counter() - t0) * 1000)
    traces = list(state.get("agent_traces", []))
    traces.append({
        "agent_name": "rca",
        "status": status,
        "latency_ms": latency_ms,
        "error": error,
    })

    logger.info(
        "RCA agent complete",
        extra={
            "num_records": rca_result.get("num_records_analyzed", 0),
            "latency_ms": latency_ms,
        },
    )

    return {
        "rca_result": rca_result,
        "sources": sources,
        "agent_traces": traces,
    }
