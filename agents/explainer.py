# =============================================================================
# agents/explainer.py — RAG-backed tutor with source citations + auto-notes
#
# Flow per turn:
#   1. Retrieve chunks via hybrid BM25 + Chroma MMR retriever
#   2. If retrieval confidence is low and a URL source exists → lazy-crawl
#      the most relevant sublink, ingest it, retry retrieval
#   3. Build personalised system prompt from LearningState profile
#   4. LLM answers with inline citations  e.g. [golangbot.com › switch]
#   5. Auto-save/merge a study note for this topic (skipped if unchanged)
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.messages import trim_messages
import config
from llm_factory import get_llm
from vectorstore.retriever import get_retriever
from vectorstore.crawler import lazy_crawl, retrieval_confidence, MIN_CONFIDENCE


# ---------------------------------------------------------------------------
# Format retrieved docs into context string with citations
# ---------------------------------------------------------------------------
def _clean_citation(src: str, page: str, topic: str) -> str:
    """
    Readable citation label from chunk metadata.
      URLs  → "golangbot.com › arrays and slices"
      Files → "notes.pdf p.3"
      Empty → topic name or "unknown"
    """
    src = str(src).strip().rstrip("/")

    if src.startswith("http://") or src.startswith("https://"):
        from urllib.parse import urlparse
        parsed = urlparse(src)
        domain = parsed.netloc.replace("www.", "")
        # Turn path into readable slug: /docs/concepts/memory → memory
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]
        slug = path_parts[-1].replace("-", " ").replace("_", " ") if path_parts else ""
        if slug:
            label = f"{domain} › {slug}"
        else:
            label = domain
        return f"[{label}]"

    # File source
    name = os.path.basename(src) or (topic or "unknown")
    if page != "":
        try:
            return f"[{name} p.{int(page)+1}]"
        except (ValueError, TypeError):
            pass
    return f"[{name}]"


def format_context(docs: list) -> str:
    parts = []
    for doc in docs:
        src   = doc.metadata.get("source", "")
        page  = doc.metadata.get("page", "")
        topic = doc.metadata.get("topic", "")
        citation = _clean_citation(src, page, topic)
        parts.append(f"{citation}\n{doc.page_content}")
    return "\n\n".join(parts)


# ── Note summarization ───────────────────────────────────────────────────────
# LLM-summarizes the explanation into a study note (max NOTE_MAX_CHARS).
# If a note already exists for this topic, merges old + new so context
# accumulates rather than overwriting. Falls back to truncation on LLM error.
NOTE_MAX_CHARS = 2000   # sidebar shows first 120 chars as preview

def summarize_note(topic: str, new_content: str, existing_note: str = "") -> str:
    try:
        summarizer = get_llm(temperature=0)
        if existing_note:
            prompt = (
                f"You maintain a running study note on '{topic}' for a learner.\n\n"
                f"Existing note:\n{existing_note}\n\n"
                f"New explanation just given:\n{new_content}\n\n"
                f"Merge these into ONE updated note (max {NOTE_MAX_CHARS} characters) "
                "that keeps the most important points from both, without repetition. "
                "Write in clear bullet points or short sections. No preamble."
            )
        else:
            prompt = (
                f"Summarize this explanation about '{topic}' into a detailed study note "
                f"(max {NOTE_MAX_CHARS} characters). Use bullet points and short sections. "
                f"Preserve all key concepts, definitions, and examples. No preamble.\n\n{new_content}"
            )
        summary = summarizer.invoke(prompt).content.strip()
        return summary[:NOTE_MAX_CHARS]
    except Exception:
        # Fallback: simple truncation if LLM summarization fails
        return new_content[:NOTE_MAX_CHARS]


# ---------------------------------------------------------------------------
# Explainer node
# ---------------------------------------------------------------------------
def explainer_node(state: dict) -> dict:
    messages = state["messages"]
    topic    = state.get("topic", "")
    user_id  = state.get("user_id", "default")
    profile  = state.get("profile", {})

    # ── Greeting shortcut — skip RAG entirely, no notes/progress pollution ──
    if topic.strip().lower() == "greeting":
        name = profile.get("name", user_id)
        greet_llm = get_llm(temperature=0.7)
        greet_response = greet_llm.invoke([
            SystemMessage(content=(
                f"You are a friendly AI learning companion greeting a user named '{name}'. "
                "Reply warmly in 1-2 short sentences and invite them to ask about a topic, "
                "take a quiz, or upload material. Keep it casual, no citations."
            ))
        ])
        return {
            "messages": [AIMessage(content=greet_response.content)],
            "context":  "",
        }

    # 1. Retrieve — scoped to this user only (user_id filter on ChromaDB metadata)
    retriever = get_retriever(user_id=user_id)
    last_user_msg = next(
        (m.content for m in reversed(messages) if hasattr(m, "type") and m.type == "human"),
        topic or "explain this topic"
    )
    docs = retriever.invoke(last_user_msg)

    # 1b. Lazy crawl — if confidence is low AND the user ingested a URL source,
    #     fetch the most relevant un-crawled sublink and retry retrieval once.
    confidence = retrieval_confidence(docs, last_user_msg)
    if confidence < MIN_CONFIDENCE:
        # Find a URL source ingested by this user (check Chroma metadata)
        try:
            from vectorstore.retriever import get_vectorstore
            vs  = get_vectorstore()
            raw = vs.get(
                where={"$and": [
                    {"user_id":   {"$eq": user_id}},
                    {"file_type": {"$eq": "url"}},
                ]},
                include=["metadatas"],
            )
            metas = raw.get("metadatas") or []
            # Find a parent URL (not already a crawled sublink)
            parent_url = next(
                (m.get("source", "") for m in metas
                 if m.get("crawled") != "true" and m.get("source", "").startswith("http")),
                None,
            )
            if parent_url:
                crawled = lazy_crawl(last_user_msg, user_id, parent_url, topic)
                if crawled:
                    # Re-run retrieval with newly ingested sublink content
                    docs = retriever.invoke(last_user_msg)
                    print(f"[explainer] Retried retrieval after lazy crawl — {len(docs)} docs")
        except Exception as e:
            print(f"[explainer] Lazy crawl skipped: {e}")

    # 2. Build context with citations
    context = format_context(docs) if docs else "No relevant documents found in knowledge base."

    # 3. Read user profile + quiz history from state
    level       = profile.get("level", "beginner")
    quiz_result = state.get("quiz_result", {})

    # Build quiz history note for system prompt if available
    quiz_note = ""
    if quiz_result and quiz_result.get("topic"):
        pct = round(quiz_result["score"] / quiz_result["total"] * 100)
        quiz_note = (
            f"\nUser's last quiz: '{quiz_result['topic']}' — "
            f"{quiz_result['score']}/{quiz_result['total']} ({pct}%). "
            "Reference this if the user asks about their last quiz or score."
        )

    # 4. Build system prompt — two modes
    if docs:
        system = SystemMessage(content=(
            f"You are a patient, clear AI tutor. The user's level is: {level}.\n\n"
            "Use the context below to answer. "
            "Always cite your sources like: 'According to [filename p.X]...'\n"
            f"{quiz_note}\n\n"
            f"Context:\n{context}"
        ))
    else:
        system = SystemMessage(content=(
            f"You are a patient, clear AI tutor. The user's level is: {level}.\n\n"
            "No documents have been uploaded yet, so answer from your general knowledge. "
            "Be clear and helpful. At the end, mention: "
            "'💡 Tip: upload a document with `python -m vectorstore.ingest --source yourfile.pdf` "
            f"for answers grounded in your own material.'\n"
            f"{quiz_note}"
        ))

    # 5. Trim messages to stay within token budget
    trimmed = trim_messages(
        list(messages),
        max_tokens=config.MAX_TOKENS_IN_CONTEXT,
        token_counter=len,
        strategy="last",
        include_system=False,
    )

    # 6. Call LLM
    llm      = get_llm()
    response = llm.invoke([system] + trimmed)

    # 7. Update progress in state (SqliteSaver persists it)
    progress = state.get("progress", {"completed": [], "weak": [], "scores": {}})
    if topic and topic not in progress.get("completed", []):
        progress = {**progress, "completed": progress.get("completed", []) + [topic]}

    # 8. Auto-save note for this topic — LLM-summarized (not blunt truncation).
    #    Skip re-summarizing if a note already exists AND the new response is
    #    very similar in length (avoids burning tokens on repeat questions).
    #    If a note already exists for this topic, merge old + new before
    #    summarizing so notes accumulate understanding instead of overwriting.
    notes = state.get("notes", {})
    if topic:
        existing_note = notes.get(topic, "")
        new_text = response.content
        # Only re-summarize if: no existing note, OR new response adds >20% more content
        should_update = (
            not existing_note
            or len(new_text) > len(existing_note) * 1.2
        )
        if should_update:
            notes = {**notes, topic: summarize_note(topic, new_text, existing_note)}
        # else: keep existing note — saves one full LLM call

    return {
        "messages": [AIMessage(content=response.content)],
        "context":  context,
        "progress": progress,
        "notes":    notes,
    }
