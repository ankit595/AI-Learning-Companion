# =============================================================================
# agents/supervisor.py — routes user intent to the right agent
#
# Uses structured output (RouteDecision) — never free text.
# Reads user profile from LearningState (SqliteSaver-persisted) to personalise routing.
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Literal
from pydantic import BaseModel
from langchain_core.messages import SystemMessage
import config
from llm_factory import get_llm_with_structured_output


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------
class RouteDecision(BaseModel):
    intent: Literal["explain", "quiz", "research", "plan", "ingest", "notes", "summarise", "code", "finish", "unclear"]
    topic:  str    # e.g. "Docker containers"
    reason: str    # why this intent was chosen


# ---------------------------------------------------------------------------
# LLM with structured output
# ---------------------------------------------------------------------------
def get_supervisor_llm():
    return get_llm_with_structured_output(RouteDecision, temperature=0)


# ---------------------------------------------------------------------------
# Supervisor node
# ---------------------------------------------------------------------------
def supervisor_node(state: dict) -> dict:
    messages  = state["messages"]
    user_id   = state.get("user_id", "default")
    step      = state.get("step_count", 0)

    # Guard: too many loops
    if step >= 10:
        return {"intent": "finish", "topic": "", "step_count": step + 1}

    # If a pending intent is set (e.g. waiting for quiz prefs reply),
    # skip LLM routing and return it directly, then clear it
    pending = state.get("pending_intent", "")
    if pending:
        return {
            "intent":         pending,
            "pending_intent": "",          # clear after use
            "step_count":     step + 1,
        }

    # Read user profile from state (persisted by SqliteSaver — not InMemoryStore,
    # profile/progress live in LearningState, see state.py)
    profile = state.get("profile", {})
    level   = profile.get("level", "")
    profile_note = f"User level: {level}." if level else ""

    system = SystemMessage(content=(
        "You are a routing supervisor for an AI learning companion. "
        "Analyse the user's latest message and decide what to do next.\n\n"
        "Intents:\n"
        "  explain  → user wants to learn or understand a topic, OR sends a casual "
        "greeting/small talk (e.g. 'hey', 'hi', 'hello', 'how are you', 'good morning') "
        "— for greetings, set topic = 'greeting'\n"
        "  quiz     → user explicitly wants to BE TESTED right now "
        "(e.g. 'quiz me', 'test me', 'give me questions', 'start a quiz')\n"
        "  research → user wants to find information or research a topic\n"
        "  plan     → user wants a study roadmap, learning plan, OR asks about their "
        "OWN progress/status (e.g. 'what is my track', 'what's my progress', "
        "'how am I doing', 'what have I learned', 'my roadmap', 'what's next for me')\n"
        "  ingest   → user wants to add/upload/ingest a file or URL\n"
        "  notes    → user wants to see/show/review/delete their saved notes "
        "(e.g. 'show my notes', 'what have I noted', 'my notes', 'review notes', "
        "'delete my Docker note', 'clear all notes', 'forget my notes on X')\n"
        "  summarise → user wants a consolidated summary of what they've learned "
        "(e.g. 'summarise everything I know about RAG', 'summarize my Docker notes', "
        "'give me an overview of what I've learned', 'sum up everything'). "
        "Set topic = the specific subject, or 'everything' if no subject given.\n"
        "  code     → user wants to run, execute, or debug code "
        "(e.g. 'run this', 'execute', 'debug', 'what does this code do', shares a code block)\n"
        "  finish   → user is done: 'thanks', 'thank you', 'ok', 'okay', 'got it', "
        "'bye', 'exit', 'quit', 'cool', 'great', 'awesome', 'nice', 'perfect', "
        "or any short acknowledgement with no question\n"
        "  unclear  → message is TRULY ambiguous with no reasonable interpretation "
        "(e.g. 'last quiz session', 'that thing', 'what about it'). "
        "Do NOT use unclear for greetings or questions about the user's own progress — "
        "those go to 'explain' or 'plan' respectively.\n\n"
        f"{profile_note}\n"
        "For 'ingest', set topic = the file path or URL the user mentioned.\n"
        "For 'unclear', set topic = your best guess at what they might mean.\n"
        "Only use 'unclear' when truly no other intent fits — prefer a best guess over unclear.\n"
        "Reply with the correct RouteDecision."
    ))

    llm      = get_supervisor_llm()
    latest = next(
        (m for m in reversed(list(messages)) if hasattr(m, "type") and m.type == "human"),
        messages[-1]
    )
    decision: RouteDecision = llm.invoke([system, latest])

    return {
        "intent":      decision.intent,
        "topic":       decision.topic,
        "step_count":  step + 1,
    }
