# =============================================================================
# agents/summariser.py — "summarise everything I know about X" agent
#
# Reads from state:
#   notes    → per-topic saved notes (already summarized by explainer)
#   progress → completed topics, weak areas, quiz scores
#   topic    → what to summarise (from supervisor); if empty, summarises everything
#
# Produces a consolidated summary combining:
#   - saved notes for the topic (or all topics if none specified)
#   - quiz performance on that topic (if available)
#   - completion/weak-area status
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import SystemMessage, AIMessage
import config
from llm_factory import get_llm


def get_summariser_llm():
    return get_llm(temperature=0.3)


def summariser_node(state: dict) -> dict:
    topic    = state.get("topic", "").strip()
    notes    = state.get("notes", {})
    progress = state.get("progress", {})

    completed = progress.get("completed", [])
    weak      = progress.get("weak", [])
    scores    = progress.get("scores", {})

    if not notes:
        return {
            "messages": [AIMessage(content=(
                "📒 You don't have any notes yet to summarise. "
                "Ask me to explain a few topics first, then come back!"
            ))]
        }

    # ── Determine scope: one topic vs everything ────────────────────────────
    if topic and topic.lower() != "everything":
        # Fuzzy match topic against saved note keys
        matched_topic = next(
            (t for t in notes if t.lower() in topic.lower() or topic.lower() in t.lower()),
            None
        )
        if not matched_topic:
            topics_list = ", ".join(notes.keys())
            return {
                "messages": [AIMessage(content=(
                    f"I don't have notes on '{topic}' yet. "
                    f"Topics I have notes on: {topics_list}"
                ))]
            }
        scoped_notes = {matched_topic: notes[matched_topic]}
    else:
        scoped_notes = notes

    # ── Build source material for the LLM ────────────────────────────────────
    note_blocks = []
    for t, content in scoped_notes.items():
        block = f"### {t}\n{content}"
        if t in scores:
            s = scores[t]
            block += f"\n(Quiz score: {s['score']}/{s['total']} — {s['percent']}%)"
        if t in weak:
            block += "\n(⚠️ Marked as a weak area)"
        note_blocks.append(block)

    source_material = "\n\n".join(note_blocks)

    system = SystemMessage(content=(
        "You are summarising a learner's study notes into one clear, well-organised "
        "summary. Use headings and bullet points. Highlight weak areas that need more "
        "review, and mention quiz performance if given. Be concise but complete.\n\n"
        f"Study material:\n{source_material}"
    ))

    llm      = get_summariser_llm()
    response = llm.invoke([system])

    scope_label = list(scoped_notes.keys())[0] if len(scoped_notes) == 1 else "everything you've learned"
    header = f"📚 **Summary — {scope_label}**\n\n"

    return {"messages": [AIMessage(content=header + response.content)]}
