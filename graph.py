# =============================================================================
# graph.py — assemble the full LangGraph
#
# Flow:
#   START → supervisor → [explainer | quizzer | planner | researcher | ingestor | notes_viewer | finisher] → END
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore

import config
from state import LearningState
from agents.supervisor  import supervisor_node
from agents.explainer   import explainer_node
from agents.ingestor    import ingestor_node
from agents.quizzer     import quizzer_node
from agents.planner     import planner_node
from agents.researcher  import researcher_node
from agents.coding      import coding_node
from agents.summariser  import summariser_node
from langchain_core.messages import AIMessage


def finisher_node(state: dict) -> dict:
    """Handles goodbye / thanks messages gracefully."""
    return {"messages": [AIMessage(content="You're welcome! Feel free to ask anytime 😊")]}


def notes_node(state: dict) -> dict:
    notes    = state.get("notes", {})
    messages = state.get("messages", [])
    last_msg = next(
        (m.content for m in reversed(messages) if hasattr(m, "type") and m.type == "human"),
        ""
    ).strip().lower()

    # ── Delete support: "delete my Docker note" / "clear all notes" ─────────
    delete_triggers = ("delete", "remove", "clear", "forget")
    wants_delete = any(t in last_msg for t in delete_triggers) and "note" in last_msg

    if wants_delete:
        # "clear all notes" / "delete all my notes"
        if any(w in last_msg for w in ("all", "everything")):
            if not notes:
                return {"messages": [AIMessage(content="📒 You don't have any notes to clear.")]}
            return {
                "messages": [AIMessage(content=f"🗑️ Cleared all {len(notes)} saved note(s).")],
                "notes":    {},
            }

        # "delete my Docker note" — match topic by substring (case-insensitive)
        matched_topic = next(
            (t for t in notes if t.lower() in last_msg or last_msg in t.lower()),
            None
        )
        if matched_topic:
            remaining = {t: c for t, c in notes.items() if t != matched_topic}
            return {
                "messages": [AIMessage(content=f"�️ Deleted note on **{matched_topic}**.")],
                "notes":    remaining,
            }
        else:
            topics_list = ", ".join(notes.keys()) if notes else "none"
            return {
                "messages": [AIMessage(content=(
                    f"I couldn't find a note matching that topic. "
                    f"Your saved topics are: {topics_list}"
                ))],
            }

    # ── Default: view all notes ──────────────────────────────────────────────
    if not notes:
        msg = (
            "�📒 You don't have any saved notes yet.\n\n"
            "Notes are automatically saved every time I explain a topic to you. "
            "Ask me to explain something first!"
        )
    else:
        lines = ["📒 **Your Saved Notes:**\n"]
        for topic, content in notes.items():
            lines.append(f"### {topic}")
            lines.append(content)          # full content — no truncation
            lines.append("")
        lines.append("_💡 Tip: say 'delete my [topic] note' or 'clear all notes' to remove them._")
        msg = "\n".join(lines)
    return {"messages": [AIMessage(content=msg)]}


def clarifier_node(state: dict) -> dict:
    topic = state.get("topic", "")
    if topic:
        msg = (
            f"I want to make sure I help you correctly! "
            f"When you mentioned **'{topic}'**, did you mean:\n"
            f"  A) Explain / learn about it\n"
            f"  B) Take a quiz on it\n"
            f"  C) Research it\n"
            f"  D) Something else — please describe what you need 😊"
        )
    else:
        msg = (
            "I'm not quite sure what you're looking for! Could you clarify? For example:\n"
            "  • 'Explain RAG to me'\n"
            "  • 'Quiz me on Docker'\n"
            "  • 'Give me a study plan for LangChain'\n"
            "  • 'Add data/notes.pdf'"
        )
    return {"messages": [AIMessage(content=msg)]}


# ---------------------------------------------------------------------------
# Router — reads intent from state, picks next node
# ---------------------------------------------------------------------------
def route(state: dict) -> str:
    """Conditional edge: reads supervisor intent → returns next node name."""
    intent = state.get("intent", "unclear")
    if intent == "explain":
        return "explainer"
    if intent == "quiz":
        return "quizzer"
    if intent == "plan":
        return "planner"
    if intent == "research":
        return "researcher"
    if intent == "ingest":
        return "ingestor"
    if intent == "notes":
        return "notes_viewer"
    if intent == "summarise":
        return "summariser"
    if intent == "code":
        return "coding"
    if intent == "finish":
        return "finisher"
    if intent == "unclear":
        return "clarifier"
    # research not built yet → explainer handles with RAG
    return "explainer"


def build_graph(checkpointer, store):
    """Compile the full LangGraph StateGraph with all agent nodes wired up."""
    builder = StateGraph(LearningState)

    # Wrap nodes to inject store into state
    def supervisor_with_store(state):
        return supervisor_node({**state, "_store": store})

    def explainer_with_store(state):
        return explainer_node({**state, "_store": store})

    def ingestor_with_store(state):
        return ingestor_node({**state, "_store": store})

    def quizzer_with_store(state):
        return quizzer_node({**state, "_store": store})

    builder.add_node("supervisor", supervisor_with_store)
    builder.add_node("explainer",  explainer_with_store)
    builder.add_node("ingestor",   ingestor_with_store)
    builder.add_node("quizzer",    quizzer_with_store)
    builder.add_node("planner",    planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("coding",     coding_node)
    builder.add_node("finisher",     finisher_node)
    builder.add_node("notes_viewer", notes_node)
    builder.add_node("summariser",   summariser_node)
    builder.add_node("clarifier",    clarifier_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route)
    builder.add_edge("explainer", END)
    builder.add_edge("ingestor",  END)
    builder.add_edge("quizzer",   END)
    builder.add_edge("planner",    END)
    builder.add_edge("researcher", END)
    builder.add_edge("coding",     END)
    builder.add_edge("finisher",     END)
    builder.add_edge("notes_viewer", END)
    builder.add_edge("summariser",   END)
    builder.add_edge("clarifier",    END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Singleton — called once from main.py
# ── Singleton — used by main.py terminal loop only ───────────────────────────
_graph = None
_store = None

def get_graph():
    """Lazy singleton for the terminal chat loop (main.py). API uses build_graph() directly."""
    global _graph, _store
    if _graph is None:
        _store = InMemoryStore()
        with SqliteSaver.from_conn_string(config.SQLITE_PATH) as checkpointer:
            _graph = build_graph(checkpointer, _store)
    return _graph, _store
