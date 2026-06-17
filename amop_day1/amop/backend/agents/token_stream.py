"""
Thread-safe registry linking a background thread to its SSE event loop + queue.

Usage pattern:
  - graph.py registers (thread_id, loop, queue) before running the LangGraph stream
  - rca_agent.py calls push_token() while streaming LLM output; the token is
    forwarded to the correct SSE queue via loop.call_soon_threadsafe()
  - graph.py unregisters on cleanup so the dict never leaks
"""

import threading

_registry: dict = {}  # thread_id -> (asyncio.AbstractEventLoop, asyncio.Queue)
_lock = threading.Lock()


def register(thread_id: int, loop, queue) -> None:
    with _lock:
        _registry[thread_id] = (loop, queue)


def unregister(thread_id: int) -> None:
    with _lock:
        _registry.pop(thread_id, None)


def push_token(agent: str, token: str) -> None:
    """Push a single LLM output token as an SSE event into the current thread's queue."""
    thread_id = threading.current_thread().ident
    with _lock:
        entry = _registry.get(thread_id)
    if entry is None:
        return
    loop, queue = entry
    event = {"type": "token", "agent": agent, "token": token}
    loop.call_soon_threadsafe(queue.put_nowait, ("event", event, {}))
