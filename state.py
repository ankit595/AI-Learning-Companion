# =============================================================================
# state.py — shared LangGraph state
# SqliteSaver persists ALL fields here per thread_id across restarts.
# No need for separate JSON/DB for user profiles — they live here.
# =============================================================================

from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class LearningState(TypedDict):
    # Full conversation history — add_messages merges on each turn
    messages:    Annotated[list[BaseMessage], add_messages]

    # Supervisor decision (reset each turn)
    intent:      str    # "explain" / "quiz" / "research" / "plan" / "ingest" / "finish"
    topic:       str    # what topic is being discussed

    # User identity
    user_id:     str

    # User profile — persisted by SqliteSaver automatically ✅
    profile:     dict   # {"name", "level", "session_count"}
    progress:    dict   # {"completed": [], "weak": [], "scores": {}}

    # Explainer — RAG retrieved chunks
    context:     str

    # Notes — auto-saved after each explanation, keyed by topic
    # e.g. {"RAG": "RAG stands for...", "Docker": "Docker is..."}
    notes:       dict

    # Quizzer — structured result
    quiz_result: dict
    quiz_prefs:  dict   # {"pending": True, "topic": ..., "num": 5, "difficulty": "medium", "subtopic": ""}

    # Quizzer — active multi-turn quiz session (one question shown per turn)
    # {"active": True, "topic", "questions": [ {question, option_a..d, correct_answer, explanation}, ... ],
    #  "current_index": 0, "score": 0, "weak_topics": [], "difficulty": "", "num_questions": 0}
    quiz_session: dict

    # Planner — pending preferences before generating a study plan
    # {"pending": True, "topic": ...}
    plan_prefs:  dict

    # Routing — if set, supervisor skips LLM and routes directly to this intent
    pending_intent: str


    # Knowledge agent — Neo4j results (Phase 5)
    graph_data:  dict

    # Loop guard
    step_count:  int
