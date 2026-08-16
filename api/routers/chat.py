# =============================================================================
# api/routers/chat.py — POST /chat
#
# Receives { user_id, message } → runs through LangGraph → returns AI response
# This is the core endpoint — the whole graph lives here.
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, AIMessage

import config
from state import LearningState
from graph import build_graph
from memory.checkpointer import get_checkpointer, get_thread_config
from langgraph.store.memory import InMemoryStore

from api.models import ChatRequest, ChatResponse
from api.dependencies import load_saved_state

router = APIRouter(prefix="/chat", tags=["chat"])

# One shared store per process
_store = InMemoryStore()


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the AI Learning Companion.
    The graph handles routing to the right agent automatically.
    """
    user_id = request.user_id.strip()
    message = request.message.strip()

    if not user_id:
        raise HTTPException(status_code=400, detail="user_id cannot be empty")
    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    thread_conf = get_thread_config(user_id)

    try:
        with get_checkpointer() as checkpointer:
            graph = build_graph(checkpointer, _store)

            # Load persisted state from previous sessions
            saved = load_saved_state(user_id)

            session_count = saved["profile"].get("session_count", 0) + 1
            profile = {
                "name":          saved["profile"].get("name", user_id),
                "level":         saved["profile"].get("level", "beginner"),
                "session_count": session_count,
            }

            initial_state: LearningState = {
                "messages":       [HumanMessage(content=message)],
                "intent":         "",
                "topic":          "",
                "user_id":        user_id,
                "profile":        profile,
                "progress":       saved["progress"],
                "context":        "",
                "notes":          saved["notes"],
                "quiz_result":    {},
                "quiz_prefs":     saved["quiz_prefs"],
                "quiz_session":   saved["quiz_session"],
                "plan_prefs":     saved.get("plan_prefs", {}),
                "pending_intent": saved["pending_intent"],
                "graph_data":     {},
                "step_count":     0,
            }

            result = graph.invoke(initial_state, thread_conf)

            # Extract last AI message
            ai_response = ""
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage):
                    ai_response = msg.content
                    break

            return ChatResponse(
                user_id  = user_id,
                response = ai_response,
                intent   = result.get("intent", ""),
                topic    = result.get("topic", ""),
                context  = result.get("context", ""),
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
