# =============================================================================
# memory/checkpointer.py — SqliteSaver setup
#
# SqliteSaver persists the full conversation state (messages, intent, topic)
# per thread_id across process restarts.
#
# thread_id = "session-{user_id}" → each user has their own conversation history
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import contextmanager
from langgraph.checkpoint.sqlite import SqliteSaver
import config


@contextmanager
def get_checkpointer():
    """Context manager — use with 'with get_checkpointer() as cp:'"""
    with SqliteSaver.from_conn_string(config.SQLITE_PATH) as checkpointer:
        yield checkpointer


def get_thread_config(user_id: str) -> dict:
    """Returns the config dict for graph.invoke() — ties session to user."""
    return {"configurable": {"thread_id": f"session-{user_id}"}}
