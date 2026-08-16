# =============================================================================
# memory/store.py — InMemoryStore setup + helpers
#
# Three namespaces:
#   ("profiles", user_id) → name, level, learning_style, session_count
#   ("progress", user_id) → completed topics, quiz scores, weak areas
#   ("notes",    user_id) → auto-saved explanation notes per topic
#
# InMemoryStore lives for the process lifetime.
# On restart: profiles are lost → user re-introduces themselves.
# (Phase 5+: replace with persistent store if needed)
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.store.memory import InMemoryStore

# Singleton store — shared across all nodes via graph.py
_store = None

def get_store() -> InMemoryStore:
    global _store
    if _store is None:
        _store = InMemoryStore()
    return _store


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------
def get_profile(store: InMemoryStore, user_id: str) -> dict:
    items = store.search(("profiles", user_id))
    return items[0].value if items else {}


def save_profile(store: InMemoryStore, user_id: str, profile: dict):
    store.put(("profiles", user_id), user_id, profile)


def init_profile(store: InMemoryStore, user_id: str) -> dict:
    """Create a default profile if none exists. Returns the profile."""
    profile = get_profile(store, user_id)
    if not profile:
        profile = {
            "name":          user_id,
            "level":         "beginner",
            "session_count": 0,
        }
        save_profile(store, user_id, profile)
    return profile


def increment_session(store: InMemoryStore, user_id: str):
    profile = get_profile(store, user_id)
    profile["session_count"] = profile.get("session_count", 0) + 1
    save_profile(store, user_id, profile)


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------
def get_progress(store: InMemoryStore, user_id: str) -> dict:
    items = store.search(("progress", user_id))
    return items[0].value if items else {"completed": [], "weak": [], "scores": {}}


def mark_completed(store: InMemoryStore, user_id: str, topic: str):
    progress = get_progress(store, user_id)
    if topic and topic not in progress["completed"]:
        progress["completed"].append(topic)
    store.put(("progress", user_id), user_id, progress)


def add_weak_area(store: InMemoryStore, user_id: str, topic: str):
    progress = get_progress(store, user_id)
    if topic and topic not in progress["weak"]:
        progress["weak"].append(topic)
    store.put(("progress", user_id), user_id, progress)


# ---------------------------------------------------------------------------
# Notes helpers
# ---------------------------------------------------------------------------
def save_note(store: InMemoryStore, user_id: str, topic: str, content: str):
    store.put(("notes", user_id), topic, {"topic": topic, "content": content})


def get_notes(store: InMemoryStore, user_id: str) -> list:
    return [item.value for item in store.search(("notes", user_id))]
