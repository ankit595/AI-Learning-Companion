# =============================================================================
# agents/coding.py — explains + runs code snippets
#
# Flow:
#   1. Extract code block from user message (```python...``` or plain text)
#   2. Detect language (Python / JS / Bash / ...)
#   3. Safety check (block dangerous patterns)
#   4. Run in sandbox (Python only currently)
#   5. LLM explains what the code does + any errors
#
# Returns: language label + explanation + stdout/stderr + exec time
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import SystemMessage, AIMessage

import config
from llm_factory import get_llm
from tools.code_runner import extract_code, run_code


# ---------------------------------------------------------------------------
# Coding node
# ---------------------------------------------------------------------------
def coding_node(state: dict) -> dict:
    messages = state["messages"]
    profile  = state.get("profile", {})
    level    = profile.get("level", "beginner")

    # Get last user message
    last_user_msg = next(
        (m.content for m in reversed(messages) if hasattr(m, "type") and m.type == "human"),
        ""
    )

    # ── 1. Extract code + language hint ──────────────────────────────────────
    code, lang_hint = extract_code(last_user_msg)

    if not code:
        return {
            "messages": [AIMessage(content=(
                "I couldn't find any code in your message.\n\n"
                "Please share code like this:\n"
                "```python\nprint('hello world')\n```"
            ))]
        }

    # ── 2. Run code ───────────────────────────────────────────────────────────
    result = run_code(code, language_hint=lang_hint)
    language  = result["language"]
    blocked   = result.get("blocked", False)
    success   = result["success"]
    stdout    = result["stdout"].strip()
    stderr    = result["stderr"].strip()
    exec_time = result["exec_time"]

    print(f"[coding] Language: {language} | success={success} | time={exec_time}s")

    # ── 3. Build execution summary ────────────────────────────────────────────
    exec_summary = f"🔍 **Language detected:** {language}\n\n"

    if blocked:
        exec_summary += f"🚫 **Blocked:** {result['reason']}\n"
    elif language != "python":
        exec_summary += f"⚠️ Running {language} is not supported yet (Python only).\n"
    elif result.get("timed_out"):
        exec_summary += "⏱️ **Timed out** after 5 seconds — possible infinite loop.\n"
    elif success:
        exec_summary += f"▶️ **Output:**\n```\n{stdout if stdout else '(no output)'}\n```\n"
        exec_summary += f"⏱️ Ran in {exec_time}s\n"
    else:
        exec_summary += f"❌ **Error:**\n```\n{stderr}\n```\n"

    # ── 4. LLM explains the code ──────────────────────────────────────────────
    explain_prompt = (
        f"The user shared this {language} code:\n\n```{language}\n{code}\n```\n\n"
    )

    if blocked:
        explain_prompt += f"It was blocked for safety: {result['reason']}. Explain why this is dangerous."
    elif success and stdout:
        explain_prompt += f"It ran successfully and produced:\n{stdout}\n\nExplain what the code does."
    elif not success and stderr:
        explain_prompt += f"It failed with:\n{stderr}\n\nExplain the error and how to fix it."
    else:
        explain_prompt += "Explain what this code does."

    system = SystemMessage(content=(
        f"You are a coding tutor. User level: {level}.\n"
        "Be concise. Structure your response as:\n"
        "1. What it does (2-3 sentences)\n"
        "2. How it works (key steps)\n"
        "3. Any issues or improvements (if applicable)\n"
        "Do NOT repeat the code back unless showing a fix."
    ))

    llm      = get_llm(temperature=0.2)
    response = llm.invoke([system, AIMessage(content=explain_prompt)])

    # ── 5. Combine execution output + explanation ─────────────────────────────
    final = f"{exec_summary}\n---\n\n💡 **Explanation:**\n{response.content}"

    return {
        "messages": [AIMessage(content=final)],
    }
