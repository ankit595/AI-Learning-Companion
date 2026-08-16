# =============================================================================
# agents/researcher.py — combines Wikipedia + user's ChromaDB docs
#
# Flow:
#   1. Search Wikipedia for the topic
#   2. Search user's ingested ChromaDB docs for the topic
#   3. Combine both into one LLM prompt
#   4. LLM synthesises a well-cited research answer
#   5. Optionally ingests Wikipedia findings into ChromaDB for future use
#
# Sources cited as:
#   [Wikipedia: <title>] for Wikipedia results
#   [<filename> | <topic>] for ChromaDB results
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import SystemMessage, AIMessage
import config
from llm_factory import get_llm
from tools.search import wikipedia_search, chroma_search


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
def get_researcher_llm():
    return get_llm(temperature=0.3)


# ---------------------------------------------------------------------------
# Researcher node
# ---------------------------------------------------------------------------
def researcher_node(state: dict) -> dict:
    topic   = state.get("topic", "")
    user_id = state.get("user_id", "default")
    messages = state["messages"]

    # Get the actual query from last user message
    # Clean filler phrases so Wikipedia gets a clean search term
    last_user_msg = next(
        (m.content for m in reversed(messages) if hasattr(m, "type") and m.type == "human"),
        topic
    )

    # Use supervisor's topic as the Wikipedia query — it's already clean (2-5 words)
    # Fall back to full message only if topic is empty
    search_query = topic if topic else last_user_msg

    print(f"[researcher] Searching Wikipedia for: '{search_query}'")
    print(f"[researcher] Searching ChromaDB for user: '{user_id}'")

    # ── 1. Search both sources ───────────────────────────────────────────────
    wiki_result   = wikipedia_search(search_query, sentences=6)
    chroma_result = chroma_search(last_user_msg, user_id=user_id, k=3)

    # ── 2. Build combined context for LLM ───────────────────────────────────
    context_parts = []

    if wiki_result["found"]:
        print(f"[researcher] Wikipedia found: '{wiki_result['title']}'")
        context_parts.append(
            f"[Wikipedia: {wiki_result['title']}] ({wiki_result['url']})\n"
            f"{wiki_result['content']}"
        )
    else:
        print(f"[researcher] Wikipedia: {wiki_result['content']}")

    if chroma_result["found"]:
        print(f"[researcher] ChromaDB: found {len(chroma_result['chunks'])} chunk(s)")
        context_parts.append(chroma_result["content"])
    else:
        print(f"[researcher] ChromaDB: {chroma_result['content']}")

    # ── 3. Handle no results ─────────────────────────────────────────────────
    if not context_parts:
        return {
            "messages": [AIMessage(content=(
                f"I couldn't find information about **{topic}** in Wikipedia "
                "or your knowledge base.\n\n"
                "💡 Try:\n"
                "- Using a more specific search term\n"
                "- Adding relevant documents: 'add data/yourfile.pdf'"
            ))],
        }

    context = "\n\n---\n\n".join(context_parts)

    # ── 4. LLM synthesises answer ────────────────────────────────────────────
    profile = state.get("profile", {})
    level   = profile.get("level", "beginner")

    system = SystemMessage(content=(
        f"You are a research assistant. The user's level is: {level}.\n\n"
        "Using the sources below, provide a clear, well-structured research summary.\n"
        "Rules:\n"
        "- Always cite your sources: 'According to [Wikipedia: Title]...' or 'From [filename]...'\n"
        "- Synthesise across sources — don't just copy-paste\n"
        "- Highlight any contradictions or gaps between sources\n"
        "- End with: '📚 Sources used:' followed by a list\n\n"
        f"Sources:\n{context}"
    ))

    llm      = get_researcher_llm()
    response = llm.invoke([system])

    # ── 5. Build sources footer ──────────────────────────────────────────────
    sources = []
    if wiki_result["found"]:
        sources.append(f"• Wikipedia: {wiki_result['title']} — {wiki_result['url']}")
    if chroma_result["found"]:
        for chunk in chroma_result["chunks"]:
            src = chunk["citation"]
            if src not in sources:
                sources.append(f"• Your docs: {src}")

    return {
        "messages": [AIMessage(content=response.content)],
        "context":  context,
    }
