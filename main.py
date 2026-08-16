# =============================================================================
# main.py — terminal entry point
# Profile + progress live in LearningState → persisted by SqliteSaver.
# No JSON file, no separate DB. Restart → user is remembered automatically.
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage

import config
from state import LearningState
from graph import build_graph
from memory.checkpointer import get_checkpointer, get_thread_config


def main():
    user_id     = input("Your name: ").strip() or "default"
    thread_conf = get_thread_config(user_id)

    with get_checkpointer() as checkpointer:
        from langgraph.store.memory import InMemoryStore
        store = InMemoryStore()
        graph = build_graph(checkpointer, store)

        # ── Read existing state from checkpointer (if any) ──────────────────
        existing = checkpointer.get(thread_conf)
        saved_profile        = {}
        saved_progress       = {}
        saved_quiz_prefs     = {}
        saved_quiz_session   = {}
        saved_plan_prefs     = {}
        saved_pending_intent = ""
        saved_notes          = {}
        if existing and existing.get("channel_values"):
            ch = existing["channel_values"]
            saved_profile        = ch.get("profile",         {})
            saved_progress       = ch.get("progress",        {})
            saved_quiz_prefs     = ch.get("quiz_prefs",      {})
            saved_quiz_session   = ch.get("quiz_session",    {})
            saved_plan_prefs     = ch.get("plan_prefs",      {})
            saved_pending_intent = ch.get("pending_intent",  "")
            saved_notes          = ch.get("notes",           {})

        # ── Init / update profile ────────────────────────────────────────────
        session_count = saved_profile.get("session_count", 0) + 1
        profile = {
            "name":          saved_profile.get("name", user_id),
            "level":         saved_profile.get("level", "beginner"),
            "session_count": session_count,
        }

        # ── Greet ────────────────────────────────────────────────────────────
        if session_count == 1:
            print(f"\n👋 Hello {user_id}! Welcome to your AI Learning Companion.")
        else:
            completed = saved_progress.get("completed", [])
            recent    = ", ".join(completed[-3:]) if completed else "nothing yet"
            print(f"\n👋 Welcome back {user_id}! Session #{session_count}.")
            print(f"   Recently studied: {recent}")
        print("   Type your question, or 'quit' to exit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "bye", "q"):
                print("👋 Goodbye!")
                break

            initial_state: LearningState = {
                "messages":       [HumanMessage(content=user_input)],
                "intent":         "",
                "topic":          "",
                "user_id":        user_id,
                "profile":        profile,
                "progress":       saved_progress,
                "context":        "",
                "notes":          saved_notes,
                "quiz_result":    {},
                "quiz_prefs":     saved_quiz_prefs,
                "quiz_session":   saved_quiz_session,
                "plan_prefs":     saved_plan_prefs,
                "pending_intent": saved_pending_intent,
                "graph_data":     {},
                "step_count":     0,
            }

            result               = graph.invoke(initial_state, config=thread_conf)
            saved_progress       = result.get("progress",       saved_progress)
            profile              = result.get("profile",        profile)
            saved_quiz_prefs     = result.get("quiz_prefs",     {})
            saved_quiz_session   = result.get("quiz_session",   {})
            saved_plan_prefs     = result.get("plan_prefs",     {})
            saved_pending_intent = result.get("pending_intent", "")
            saved_notes          = result.get("notes",          saved_notes)

            messages = result.get("messages", [])
            for msg in reversed(messages):
                if hasattr(msg, "type") and msg.type == "ai":
                    print(f"\n🤖 {msg.content}\n")
                    break


if __name__ == "__main__":
    main()
