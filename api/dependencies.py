# =============================================================================
# api/dependencies.py — Shared dependencies injected into routers via Depends()
#
# Why here? Multiple endpoints need the graph + checkpointer.
# Define once, inject everywhere — classic DI pattern.
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.store.memory import InMemoryStore
from memory.checkpointer import get_checkpointer, get_thread_config
from graph import build_graph

# ── Singleton store (shared across all requests in this process) ───────────────
_store = InMemoryStore()


def get_graph_and_checkpointer():
    """
    Returns (graph, checkpointer) — checkpointer is a context manager,
    so callers must use it inside a `with` block.
    Used by /chat endpoint which needs both.
    """
    return _store


def get_store() -> InMemoryStore:
    """Return the shared in-memory store."""
    return _store


def load_saved_state(user_id: str) -> dict:
    """
    Read persisted state for a user from SqliteSaver.
    Returns dict with profile, progress, notes, quiz_prefs, pending_intent.
    Used by /chat, /notes, /profile endpoints.
    """
    thread_conf = get_thread_config(user_id)
    with get_checkpointer() as checkpointer:
        existing = checkpointer.get(thread_conf)
        if existing and existing.get("channel_values"):
            ch = existing["channel_values"]
            return {
                "profile":        ch.get("profile",        {}),
                "progress":       ch.get("progress",       {}),
                "notes":          ch.get("notes",          {}),
                "quiz_prefs":     ch.get("quiz_prefs",     {}),
                "quiz_session":   ch.get("quiz_session",   {}),
                "plan_prefs":     ch.get("plan_prefs",     {}),
                "pending_intent": ch.get("pending_intent", ""),
            }
    return {
        "profile": {}, "progress": {}, "notes": {},
        "quiz_prefs": {}, "quiz_session": {}, "plan_prefs": {}, "pending_intent": ""
    }
