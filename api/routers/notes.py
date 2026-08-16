# =============================================================================
# api/routers/notes.py — GET /notes/{user_id}
#
# Returns all saved notes for a user from persisted LearningState.
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException
from api.models import NotesResponse
from api.dependencies import load_saved_state

router = APIRouter(prefix="/notes", tags=["notes"])


@router.get("/{user_id}", response_model=NotesResponse)
def get_notes(user_id: str):
    """
    Fetch all saved notes for a user.
    Notes are auto-saved by the explainer agent after each explanation.
    """
    try:
        saved = load_saved_state(user_id)
        return NotesResponse(
            user_id = user_id,
            notes   = saved["notes"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
