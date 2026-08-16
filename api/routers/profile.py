# =============================================================================
# api/routers/profile.py — GET /profile/{user_id}
#
# Returns user profile + progress from persisted LearningState.
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException
from api.models import ProfileResponse
from api.dependencies import load_saved_state

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get("/{user_id}", response_model=ProfileResponse)
def get_profile(user_id: str):
    """
    Fetch user profile and progress.
    Streamlit sidebar uses this to show stats, completed topics, weak areas.
    """
    try:
        saved = load_saved_state(user_id)
        return ProfileResponse(
            user_id  = user_id,
            profile  = saved["profile"],
            progress = saved["progress"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
