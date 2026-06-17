"""
OpenRouter client — OpenAI-compatible SDK pointed at openrouter.ai.

API key rotation: if OPENROUTER_API_KEYS is set (comma-separated list),
keys are tried in shuffled order on each call for load distribution.
On RateLimitError, the next key in the shuffled sequence is tried so a
single exhausted key never blocks the pipeline. Only after all keys are
rate-limited for a given model does the error propagate to the caller's
model-fallback loop.
"""

import random

from openai import OpenAI, RateLimitError

from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


def _get_key_pool() -> list[str]:
    """Return the list of API keys, falling back to the single primary key."""
    settings = get_settings()
    raw = settings.openrouter_api_keys.strip()
    if raw:
        pool = [k.strip() for k in raw.split(",") if k.strip()]
        if pool:
            return pool
    return [settings.openrouter_api_key]


def _make_client(api_key: str) -> OpenAI:
    settings = get_settings()
    return OpenAI(
        api_key=api_key,
        base_url=settings.openrouter_base_url,
        max_retries=0,  # disable SDK retries — key rotation in call_with_key_rotation handles this
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "AMOP - Autonomous Maintenance Operations Platform",
        },
    )


def get_openrouter_client() -> OpenAI:
    """Create a client with a randomly selected API key (for one-off use)."""
    return _make_client(random.choice(_get_key_pool()))


def stream_with_key_rotation(
    model: str,
    messages: list[dict],
    token_callback=None,
    **kwargs,
) -> str:
    """
    Like call_with_key_rotation but streams tokens from the LLM.

    Each token delta is passed to token_callback(token: str) as it arrives,
    enabling real-time SSE forwarding. Returns the full accumulated content
    string when done — identical contract to call_with_key_rotation.

    Falls back to non-streaming call_with_key_rotation if the server does not
    support streaming or if stream=True raises an unexpected error.
    """
    keys = _get_key_pool()
    shuffled = list(keys)
    random.shuffle(shuffled)

    last_exc: RateLimitError | None = None
    for i, key in enumerate(shuffled):
        try:
            client = _make_client(key)
            stream = client.chat.completions.create(
                model=model, messages=messages, stream=True, **kwargs
            )
            content = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    content += delta
                    if token_callback is not None:
                        token_callback(delta)
            if not content:
                raise ValueError(f"Model '{model}' returned empty content")
            logger.debug("Streaming LLM call succeeded", extra={"model": model})
            return content.strip()
        except RateLimitError as exc:
            logger.warning(
                "API key rate-limited (stream), rotating",
                extra={"model": model, "key_index": i},
            )
            last_exc = exc
            continue

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"All keys exhausted for model '{model}'")


def call_with_key_rotation(model: str, messages: list[dict], **kwargs) -> str:
    """
    Make a single LLM call, cycling through every API key on RateLimitError.

    Keys are shuffled before each call so load is distributed, but all keys
    are tried before giving up on a model. Non-rate-limit errors (BadRequestError,
    network failures, empty content) propagate immediately so the caller's
    model-fallback loop can move to the next model without wasting key attempts.

    Raises RateLimitError only after all keys are exhausted for this model.
    """
    keys = _get_key_pool()
    shuffled = list(keys)
    random.shuffle(shuffled)

    last_exc: RateLimitError | None = None
    for i, key in enumerate(shuffled):
        try:
            client = _make_client(key)
            response = client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError(f"Model '{model}' returned empty content")
            logger.debug("LLM call succeeded", extra={"model": model})
            return content.strip()
        except RateLimitError as exc:
            logger.warning(
                "API key rate-limited, rotating to next key",
                extra={"model": model, "key_index": i, "pool_size": len(shuffled)},
            )
            last_exc = exc
            continue
        # BadRequestError, ValueError (empty content), network errors — propagate immediately

    logger.warning(
        "All API keys rate-limited for model",
        extra={"model": model, "pool_size": len(shuffled)},
    )
    raise last_exc  # type: ignore[misc]
