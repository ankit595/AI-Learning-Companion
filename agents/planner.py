# =============================================================================
# agents/planner.py — personalised, interactive study roadmap generator
#
# Flow (two-phase turn-based, same pattern as quizzer.py):
#   Phase 1: ask the user for planning preferences (weeks, hours/week, focus)
#   Phase 2: user replies with prefs → generate personalised roadmap
#
# Reads from state:
#   profile  → name, level, session_count
#   progress → completed topics, weak areas, quiz scores
#   topic    → what to build the plan around (from supervisor)
#
# Generates:
#   A study roadmap personalised to the user's level, history, AND explicit
#   preferences (duration, weekly time budget, focus areas)
#   Skips already-completed topics
#   Re-schedules weak areas (low quiz scores)
#   Suggests next quiz topic
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import SystemMessage, AIMessage
import config
from llm_factory import get_llm


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
def get_planner_llm():
    return get_llm(temperature=0.4)


# ---------------------------------------------------------------------------
# Parse a free-text prefs reply into (weeks, hours_per_week, focus)
# Accepts things like: "4 weeks, 5 hours, focus on transformers"
# or just "4, 5" or blank (all defaults)
# ---------------------------------------------------------------------------
def parse_plan_prefs(text: str) -> dict:
    weeks         = 2
    hours_per_wk  = 5
    focus         = ""

    parts = [p.strip() for p in text.replace(";", ",").split(",")]
    numbers_seen = 0

    for part in parts:
        pl = part.lower()
        if not pl or pl in ("skip", "no", "none", "any", "default", "defaults"):
            continue
        digits = "".join(c for c in part if c.isdigit())
        if digits and numbers_seen == 0:
            weeks = max(1, min(12, int(digits)))
            numbers_seen += 1
        elif digits and numbers_seen == 1:
            hours_per_wk = max(1, min(40, int(digits)))
            numbers_seen += 1
        elif part:
            focus = part

    return {"weeks": weeks, "hours_per_week": hours_per_wk, "focus": focus}


# ---------------------------------------------------------------------------
# Planner node — two-phase turn-based flow (no blocking input()):
#   Phase 1: no prefs yet → ask the user (weeks / hours per week / focus)
#   Phase 2: prefs reply received → parse → generate personalised roadmap
# ---------------------------------------------------------------------------
def planner_node(state: dict) -> dict:
    topic       = state.get("topic", "the requested subject")
    profile     = state.get("profile",  {})
    progress    = state.get("progress", {})
    messages    = state.get("messages", [])
    plan_prefs  = state.get("plan_prefs", {})

    last_user_msg = next(
        (m.content for m in reversed(messages) if hasattr(m, "type") and m.type == "human"),
        ""
    )

    # ── PHASE 1: No prefs yet → ask the user ────────────────────────────────
    if not plan_prefs.get("pending"):
        ask_msg = (
            f"🗓️ Let's build your study plan for **{topic}**!\n\n"
            "Please answer (or press Enter to use defaults):\n"
            "1. How many weeks should the plan cover? (default: **2**)\n"
            "2. How many hours per week can you study? (default: **5**)\n"
            "3. Anything specific to focus on? (default: general coverage)\n\n"
            "Example reply: `4 weeks, 6 hours, focus on transformers` "
            "or just press Enter for defaults."
        )
        return {
            "messages":       [AIMessage(content=ask_msg)],
            "plan_prefs":     {"pending": True, "topic": topic},
            "pending_intent": "plan",   # supervisor will re-route next reply here
        }

    # ── PHASE 2: Prefs received → parse → generate personalised roadmap ────
    prefs         = parse_plan_prefs(last_user_msg)
    weeks         = prefs["weeks"]
    hours_per_wk  = prefs["hours_per_week"]
    focus         = prefs["focus"]

    # Use topic captured in phase 1, not current state (may have drifted)
    plan_topic = plan_prefs.get("topic", topic)

    # ── Extract user context ─────────────────────────────────────────────────
    name          = profile.get("name",          "Learner")
    level         = profile.get("level",         "beginner")
    session_count = profile.get("session_count", 1)

    completed = progress.get("completed", [])
    weak      = progress.get("weak",      [])
    scores    = progress.get("scores",    {})

    # ── Build context strings for the prompt ─────────────────────────────────
    completed_str = ", ".join(completed) if completed else "none yet"

    weak_str = ""
    if weak:
        weak_details = []
        for w in weak:
            score_info = scores.get(w, {})
            if score_info:
                weak_details.append(
                    f"{w} (scored {score_info.get('score', '?')}/"
                    f"{score_info.get('total', '?')} = "
                    f"{score_info.get('percent', '?')}%)"
                )
            else:
                weak_details.append(w)
        weak_str = ", ".join(weak_details)

    # Find next quiz suggestion — lowest scoring topic
    next_quiz = ""
    if scores:
        lowest = min(scores.items(), key=lambda x: x[1].get("percent", 100))
        if lowest[1].get("percent", 100) < 80:
            next_quiz = lowest[0]

    focus_line = f"\n  Specific focus requested: {focus}" if focus else ""

    # ── System prompt ────────────────────────────────────────────────────────
    system = SystemMessage(content=(
        f"You are a personalised study planner for an AI learning companion.\n\n"
        f"User profile:\n"
        f"  Name: {name}\n"
        f"  Level: {level}\n"
        f"  Sessions completed: {session_count}\n\n"
        f"Learning history:\n"
        f"  Topics already covered: {completed_str}\n"
        f"  Weak areas (low quiz scores): {weak_str if weak_str else 'none identified yet'}\n"
        f"  {'Suggested quiz: ' + next_quiz if next_quiz else ''}\n\n"
        f"Planning preferences (explicitly requested by the user):\n"
        f"  Duration: {weeks} week(s)\n"
        f"  Time budget: {hours_per_wk} hour(s) per week"
        f"{focus_line}\n\n"
        f"Task: Create a clear, personalised study plan for: '{plan_topic}' "
        f"spanning EXACTLY {weeks} week(s), sized to fit {hours_per_wk} hour(s)/week.\n\n"
        f"Rules:\n"
        f"- Adjust depth/pace to the user's level ({level}): "
        f"beginners need more foundational steps, advanced users can move faster\n"
        f"- Skip topics already in 'covered' list — mention them as prerequisites done\n"
        f"- Re-schedule weak areas in the plan for revision\n"
        f"- Break into exactly {weeks} week(s), each with daily/session-level focus areas "
        f"that realistically fit within {hours_per_wk} hour(s) per week\n"
        f"- If a specific focus area was requested, prioritise it throughout the plan\n"
        f"- End with: 'Suggested next quiz: X' (based on weak areas or next logical topic)\n"
        f"- Use emojis for readability, keep it motivating\n"
        f"- Be specific — name actual subtopics, not just 'learn more about X'\n"
    ))

    llm      = get_planner_llm()
    response = llm.invoke([system])

    return {
        "messages":       [AIMessage(content=response.content)],
        "plan_prefs":     {},   # clear — plan generated
        "pending_intent": "",   # release control back to supervisor
    }
