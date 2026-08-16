# =============================================================================
# agents/ingestor.py — handles user requests to add files/URLs to knowledge base
# Triggered when supervisor detects intent = "ingest"
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from langchain_core.messages import AIMessage

import config
from vectorstore.ingest import ingest

BASE_DIR = config.BASE_DIR


# ---------------------------------------------------------------------------
# Resolve path — tries relative to BASE_DIR first, then absolute
# ---------------------------------------------------------------------------
def resolve_path(source: str) -> str:
    if source.startswith("http"):
        return source
    from pathlib import Path
    p = Path(source)
    if p.is_absolute() and p.exists():
        return str(p)
    # Try relative to project root
    resolved = BASE_DIR / p
    if resolved.exists():
        return str(resolved)
    # Try relative to data/
    resolved2 = BASE_DIR / "data" / p.name
    if resolved2.exists():
        return str(resolved2)
    return str(p)  # return as-is, will fail with clear error


# ---------------------------------------------------------------------------
# Extract file path or URL from user message
# e.g. "add data/rag_notes.txt" → "data/rag_notes.txt"
#      "ingest https://youtube.com/..." → "https://youtube.com/..."
# ---------------------------------------------------------------------------
def extract_source(message: str) -> str:
    # URL
    url = re.search(r'https?://\S+', message)
    if url:
        return url.group()

    # File path (with extension or data/ prefix)
    path = re.search(r'[\w./\-]+\.(?:pdf|txt|csv|json|md|rst)', message)
    if path:
        return path.group()

    # data/ prefix without known extension
    data = re.search(r'data/[\w./\-]+', message)
    if data:
        return data.group()

    return ""


# ---------------------------------------------------------------------------
# Ingestor node
# ---------------------------------------------------------------------------
def ingestor_node(state: dict) -> dict:
    messages = state["messages"]
    topic    = state.get("topic", "")   # supervisor may have set this

    # Get last user message
    last_user_msg = next(
        (m.content for m in reversed(messages) if hasattr(m, "type") and m.type == "human"),
        ""
    )

    # Always extract source from the current user message only
    # Never use state["topic"] — it may be stale from checkpointer
    source = extract_source(last_user_msg)

    # Fallback: if supervisor put a clean path in topic AND it's not in messages yet
    if not source:
        t = state.get("topic", "")
        if t and ("/" in t or "http" in t or "." in t):
            source = resolve_path(t)

    if not source:
        return {
            "messages": [AIMessage(content=(
                "I couldn't find a file path or URL in your message. "
                "Try: 'add data/myfile.pdf' or 'ingest https://...'"
            ))]
        }

    # Resolve to absolute path
    source = resolve_path(source)

    # Run ingest
    user_id = state.get("user_id", "shared")
    try:
        n = ingest(source, user_id=user_id)   # topic auto-derived inside ingest()
        return {
            "messages": [AIMessage(content=(
                f"✅ Done! Ingested **{source}** into the knowledge base.\n"
                f"   {n} chunks stored. You can now ask questions about it."
            ))]
        }
    except Exception as e:
        return {
            "messages": [AIMessage(content=(
                f"❌ Failed to ingest `{source}`.\n"
                f"Error: {e}\n"
                "Make sure the file exists in the `data/` folder."
            ))]
        }
