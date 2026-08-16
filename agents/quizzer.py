# =============================================================================
# agents/quizzer.py — generates MCQ quiz from ingested docs + tracks score
#
# Flow:
#   1. Retrieve relevant chunks from ChromaDB (scoped to user)
#   2. LLM generates N questions as structured output (QuizSet)
#   3. User answers each question in terminal
#   4. Score calculated → saved to state (progress["scores"], progress["weak"])
#
# Structured output schema:
#   QuizQuestion: question, options (A-D), correct_answer, explanation
#   QuizSet:      topic, questions: list[QuizQuestion]
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, AIMessage
import config
from llm_factory import get_llm, get_llm_with_structured_output
from vectorstore.retriever import get_retriever


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------
class QuizQuestion(BaseModel):
    question:       str
    option_a:       str
    option_b:       str
    option_c:       str
    option_d:       str
    correct_answer: str   # "A" / "B" / "C" / "D"
    explanation:    str   # why the correct answer is right


class QuizSet(BaseModel):
    topic:     str
    questions: list[QuizQuestion]


# ---------------------------------------------------------------------------
# LLM with structured output
# ---------------------------------------------------------------------------
def get_quiz_llm():
    return get_llm_with_structured_output(QuizSet, temperature=0.7)


# ---------------------------------------------------------------------------
# Format retrieved docs into context
# ---------------------------------------------------------------------------
def format_context(docs: list) -> str:
    parts = []
    for doc in docs:
        src   = os.path.basename(str(doc.metadata.get("source", "unknown")))
        topic = doc.metadata.get("topic", "")
        parts.append(f"[{src} | {topic}]\n{doc.page_content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Fix LLM position bias — shuffle A/B/C/D options for one question dict
# ---------------------------------------------------------------------------
def shuffle_options(q: dict) -> dict:
    """
    Takes a question dict with option_a..option_d + correct_answer,
    randomly reassigns which letter each option sits under, and
    updates correct_answer to match the new position.
    """
    letters = ["A", "B", "C", "D"]
    original_correct = q["correct_answer"].strip().upper()

    # Map letter -> option text
    option_text = {
        "A": q["option_a"], "B": q["option_b"],
        "C": q["option_c"], "D": q["option_d"],
    }

    shuffled_letters = letters[:]
    random.shuffle(shuffled_letters)

    new_q = dict(q)
    for new_letter, old_letter in zip(letters, shuffled_letters):
        new_q[f"option_{new_letter.lower()}"] = option_text[old_letter]
        if old_letter == original_correct:
            new_q["correct_answer"] = new_letter

    return new_q


# ---------------------------------------------------------------------------
# Format one question as a chat message (markdown)
# ---------------------------------------------------------------------------
def format_question_msg(q: dict, index: int, total: int) -> str:
    return (
        f"**Question {index + 1} / {total}**\n\n"
        f"{q['question']}\n\n"
        f"- **A)** {q['option_a']}\n"
        f"- **B)** {q['option_b']}\n"
        f"- **C)** {q['option_c']}\n"
        f"- **D)** {q['option_d']}\n\n"
        f"_Reply with A, B, C, or D._"
    )


# ---------------------------------------------------------------------------
# Parse a free-text answer into A/B/C/D (or None if unrecognised)
# ---------------------------------------------------------------------------
def parse_answer(text: str) -> str:
    t = text.strip().upper()
    for letter in ("A", "B", "C", "D"):
        if t == letter or t.startswith(f"{letter})") or t.startswith(f"{letter}."):
            return letter
    # Fall back: first standalone A-D character in the text
    for ch in t:
        if ch in ("A", "B", "C", "D"):
            return ch
    return ""


# ---------------------------------------------------------------------------
# Quizzer node — three-phase turn-based flow (no blocking input()):
#   Phase 1: ask the user for quiz preferences (num, difficulty, subtopic)
#   Phase 2: user replies with prefs → generate quiz → show question 1
#   Phase 3: user replies with an answer → grade → show next question
#            (repeats until all questions answered → final summary)
# ---------------------------------------------------------------------------
def quizzer_node(state: dict) -> dict:
    topic         = state.get("topic", "the topic")
    user_id       = state.get("user_id", "default")
    messages      = state["messages"]
    quiz_prefs    = state.get("quiz_prefs", {})
    quiz_session  = state.get("quiz_session", {})

    last_user_msg = next(
        (m.content for m in reversed(messages) if hasattr(m, "type") and m.type == "human"),
        ""
    )

    # ── PHASE 3: A quiz is already active → this message is an answer ───────
    if quiz_session.get("active"):
        questions = quiz_session["questions"]
        idx       = quiz_session["current_index"]
        q         = questions[idx]

        user_answer = parse_answer(last_user_msg)
        correct     = q["correct_answer"].strip().upper()
        is_correct  = user_answer == correct

        feedback = f"{'✅ **Correct!**' if is_correct else f'❌ **Wrong.** Correct answer: **{correct}**'}\n\n💡 {q['explanation']}"

        score       = quiz_session["score"] + (1 if is_correct else 0)
        weak_topics = quiz_session["weak_topics"]
        if not is_correct and quiz_session["topic"] not in weak_topics:
            weak_topics = weak_topics + [quiz_session["topic"]]

        next_idx = idx + 1
        total    = len(questions)

        if next_idx < total:
            # More questions remain → show feedback + next question, stay in quiz mode
            next_q_msg = format_question_msg(questions[next_idx], next_idx, total)
            combined   = f"{feedback}\n\n---\n\n{next_q_msg}"
            return {
                "messages":       [AIMessage(content=combined)],
                "quiz_session":   {**quiz_session, "current_index": next_idx, "score": score, "weak_topics": weak_topics},
                "pending_intent": "quiz",   # keep routing to quizzer for the next answer
            }

        # ── Quiz finished — compute final results ───────────────────────────
        percentage = round((score / total) * 100)
        if percentage >= 80:
            verdict = "🌟 Excellent! You've got this topic well covered."
        elif percentage >= 50:
            verdict = "👍 Good effort! Review the explanations above to strengthen weak spots."
        else:
            verdict = "📚 Keep studying! Re-read the material and try again."

        progress = state.get("progress", {"completed": [], "weak": [], "scores": {}})
        scores   = progress.get("scores", {})
        scores[quiz_session["topic"]] = {"score": score, "total": total, "percent": percentage}
        weak = progress.get("weak", [])
        for w in weak_topics:
            if w not in weak:
                weak.append(w)
        progress = {**progress, "scores": scores, "weak": weak}

        summary = (
            f"{feedback}\n\n---\n\n"
            f"🏆 **Quiz complete — {quiz_session['topic']}**\n\n"
            f"**Score: {score}/{total} ({percentage}%)**\n\n"
            f"{verdict}"
        )
        if weak_topics:
            summary += f"\n\n📌 Topics to review: {', '.join(set(weak_topics))}"

        return {
            "messages":       [AIMessage(content=summary)],
            "quiz_result":    {"topic": quiz_session["topic"], "score": score, "total": total},
            "quiz_session":   {},          # clear — quiz finished
            "quiz_prefs":     {},
            "progress":       progress,
            "pending_intent": "",          # release control back to supervisor
        }

    # ── PHASE 1: No prefs yet → ask the user ────────────────────────────────
    if not quiz_prefs.get("pending"):
        ask_msg = (
            f"📋 Let's set up your quiz on **{topic}**!\n\n"
            "Please answer (or press Enter to use defaults):\n"
            "1. How many questions? (default: **5**)\n"
            "2. Difficulty? easy / medium / hard (default: **medium**)\n"
            "3. Specific subtopic to focus on? (default: **general**)\n\n"
            "Example reply: `10, hard, vector search`  or just press Enter for defaults."
        )
        return {
            "messages":       [AIMessage(content=ask_msg)],
            "quiz_prefs":     {"pending": True, "topic": topic},
            "pending_intent": "quiz",    # supervisor will re-route next reply here
        }

    # ── PHASE 2: Prefs received → parse → generate quiz → show question 1 ──
    num_questions = 5
    difficulty    = "medium"
    subtopic      = ""

    parts = [p.strip() for p in last_user_msg.replace(";", ",").split(",")]

    for part in parts:
        pl = part.lower()
        if pl in ("easy", "medium", "hard"):
            difficulty = pl
        elif pl in ("general", "skip", "any", "no", "none", ""):
            pass   # keep subtopic empty
        else:
            digits = "".join(c for c in part if c.isdigit())
            if digits:
                num_questions = max(1, min(20, int(digits)))
            elif part:
                subtopic = part   # treat as subtopic text

    # Use topic from prefs (set in phase 1), not from current state which may have changed
    quiz_topic = quiz_prefs.get("topic", topic)

    # Retrieve relevant chunks
    retriever = get_retriever(user_id=user_id)
    query     = f"{quiz_topic} {subtopic}".strip()
    docs      = retriever.invoke(query)
    context   = format_context(docs) if docs else ""

    difficulty_guide = {
        "easy":   "basic definitions and concepts, suitable for beginners",
        "medium": "application and understanding, mix of definition and reasoning",
        "hard":   "advanced reasoning, edge cases, comparisons and deep understanding",
    }
    focus = f" Focus specifically on: {subtopic}." if subtopic else ""

    base_rules = (
        f"You are a quiz generator. Generate exactly {num_questions} multiple-choice "
        f"questions on '{quiz_topic}'.{focus}\n"
        f"Difficulty: {difficulty} — {difficulty_guide[difficulty]}.\n\n"
        "Rules:\n"
        "- Each question must have 4 options (A, B, C, D)\n"
        "- correct_answer must be exactly 'A', 'B', 'C', or 'D'\n"
        "- Include a clear explanation for the correct answer\n"
        "- Vary question types: definitions, applications, comparisons\n\n"
    )
    system_content = (
        base_rules + f"Base questions on this content:\n{context}"
        if context else
        base_rules + "Use your general knowledge — no documents found for this topic."
    )

    llm  = get_quiz_llm()
    quiz: QuizSet = llm.invoke([SystemMessage(content=system_content)])

    # Serialize questions to plain dicts (state must stay JSON/pickle-friendly)
    questions = [q.model_dump() for q in quiz.questions]

    # ── Fix LLM position bias ────────────────────────────────────────────────
    # Structured-output LLMs tend to place the correct answer in option A far
    # more often than chance. Shuffle each question's options so the correct
    # answer lands in a random position — keeps the quiz statistically fair.
    questions = [shuffle_options(q) for q in questions]

    intro_msg  = f"📝 **Quiz: {quiz.topic}**  ({len(questions)} questions, {difficulty} difficulty)\n\n---\n\n"
    first_q    = format_question_msg(questions[0], 0, len(questions))

    return {
        "messages":     [AIMessage(content=intro_msg + first_q)],
        "quiz_prefs":   {},   # phase 1/2 prefs done, no longer needed
        "quiz_session": {
            "active":        True,
            "topic":         quiz.topic,
            "questions":     questions,
            "current_index": 0,
            "score":         0,
            "weak_topics":   [],
            "difficulty":    difficulty,
        },
        "pending_intent": "quiz",   # keep routing to quizzer for the answer
    }
