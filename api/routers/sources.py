# =============================================================================
# api/routers/sources.py — GET /sources/{user_id}
#
# Lists all previously ingested sources for a user, read directly from
# ChromaDB (which persists data permanently on disk). This lets the UI
# restore the "Sources" sidebar across Streamlit reloads/restarts —
# the data was never actually lost, only the in-memory session state was.
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from vectorstore.retriever import list_user_sources, count_user_chunks, delete_user_source

router = APIRouter(prefix="/sources", tags=["sources"])


class SourcesResponse(BaseModel):
    user_id:        str
    sources:        list   # [{"name", "type", "topic", "chunks"}, ...]
    total_chunks:   int


@router.get("/{user_id}", response_model=SourcesResponse)
def get_sources(user_id: str):
    """
    Fetch all sources previously ingested by this user (persists across
    sessions/restarts since it reads straight from ChromaDB on disk).
    """
    try:
        sources      = list_user_sources(user_id)
        total_chunks = count_user_chunks(user_id)
        return SourcesResponse(
            user_id      = user_id,
            sources      = sources,
            total_chunks = total_chunks,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DeleteSourceResponse(BaseModel):
    user_id:        str
    source_name:    str
    chunks_deleted: int
    message:        str


@router.delete("/{user_id}/{source_name:path}", response_model=DeleteSourceResponse)
def delete_source(user_id: str, source_name: str):
    """
    Delete all chunks for a specific ingested source belonging to this user.
    source_name may contain slashes (e.g. "golangbot.com/switch") — the :path
    modifier tells FastAPI to capture the full remainder as one parameter.
    """
    try:
        deleted = delete_user_source(user_id, source_name)
        if deleted == 0:
            raise HTTPException(status_code=404, detail=f"Source '{source_name}' not found for user '{user_id}'")
        return DeleteSourceResponse(
            user_id        = user_id,
            source_name    = source_name,
            chunks_deleted = deleted,
            message        = f"Deleted {deleted} chunk(s) for '{source_name}'",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
