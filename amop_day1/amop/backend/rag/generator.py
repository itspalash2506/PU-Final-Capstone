"""
LLM answer generation grounded in retrieved context.

Accuracy measures:
- temperature=0.2  → low randomness, factual recall
- Explicit instruction to cite machine IDs and say "I don't know" when context is thin
- Context truncated per doc to avoid drowning signal in a single verbose record
- Numbered source list so the model can reference [1], [2] naturally
"""

from openai import RateLimitError, BadRequestError, NotFoundError

from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.clients.openrouter import call_with_key_rotation

logger = get_logger(__name__)

_MAX_NOTE_CHARS = 400   # truncation limit per technician note
_MAX_CTX_CHARS = 3500   # total context budget (keeps prompt under token limits)

_SYSTEM_PROMPT = """\
You are an AI maintenance assistant for an industrial operations platform (AMOP).
Your role is to help technicians diagnose equipment issues using historical work order data.

Rules you MUST follow:
1. Answer in 3–5 sentences maximum. Stop as soon as the key facts are covered — do not pad.
2. Answer using the numbered context records provided. Do not invent facts not present in the records.
3. Always cite the machine ID (e.g. "Machine A81") and record number (e.g. "[1]") when referencing a record.
4. Even if the records are from different machines than the one asked about, extract any useful patterns,
   common causes, or technician actions visible in the context. Note which machines the data comes from.
5. Only say "I don't have sufficient historical data to answer this question" if the context records have
   ZERO relevance to the question (completely different equipment type or issue category).
6. Never fabricate part numbers, error codes, or procedures not present in the context.\
"""

_USER_TEMPLATE = """\
Context records (use ALL records that are relevant — cite machine IDs and record numbers):
{context}

Technician question: {question}

Answer (based only on the records above):\
"""


def _build_context(docs: list[dict]) -> str:
    """Format retrieved docs as a numbered list, truncating long notes."""
    parts: list[str] = []
    total = 0

    for i, doc in enumerate(docs, start=1):
        notes = doc.get("technician_notes") or "No technician notes recorded."
        if len(notes) > _MAX_NOTE_CHARS:
            notes = notes[:_MAX_NOTE_CHARS] + "…"

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


def _call_llm(model: str, messages: list[dict]) -> str:
    """Single LLM call with automatic key rotation on rate limit."""
    return call_with_key_rotation(model, messages, temperature=0.2)


def generate(query: str, docs: list[dict]) -> str:
    """
    Generate a grounded answer for the technician's query.
    On 429 rate-limit errors, automatically retries with fallback models
    rather than waiting — avoids 30+ second hangs on the free tier.
    """
    settings = get_settings()
    context = _build_context(docs)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(context=context, question=query)},
    ]

    fallbacks = [m.strip() for m in settings.openrouter_fallback_models.split(",") if m.strip()]
    models_to_try = [settings.openrouter_model] + fallbacks

    last_exc: Exception | None = None
    for model in models_to_try:
        try:
            answer = _call_llm(model, messages)
            logger.info("LLM generation successful", extra={"model": model, "answer_len": len(answer)})
            return answer
        except (RateLimitError, BadRequestError, NotFoundError, ValueError) as exc:
            # RateLimitError → upstream throttling
            # BadRequestError → invalid/retired model ID
            # ValueError → empty content
            # All are skippable — try next model immediately.
            logger.warning("Model unavailable, trying fallback", extra={"model": model, "error": str(exc)})
            last_exc = exc
            continue
        except Exception as exc:
            logger.error("LLM generation failed", extra={"model": model, "error": str(exc)}, exc_info=True)
            last_exc = exc
            break

    if settings.app_env == "development":
        return f"[LLM ERROR — {type(last_exc).__name__}]: {last_exc}"
    return (
        "I was unable to generate an answer due to a service error. "
        "Please check connectivity and try again."
    )
