# =============================================================================
# api/models.py — All Pydantic request/response schemas for the FastAPI layer
# =============================================================================

from pydantic import BaseModel
from typing import Optional


# ── /chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    user_id:  str
    response: str          # last AI message
    intent:   str          # what the supervisor decided
    topic:    str          # what topic was discussed
    context:  Optional[str] = ""   # retrieved RAG chunks (raw text, if any)


# ── /ingest ───────────────────────────────────────────────────────────────────

class IngestURLRequest(BaseModel):
    user_id: str
    url:     str
    topic:   Optional[str] = ""   # optional — ingest() can derive it

class IngestResponse(BaseModel):
    status:  str           # "ok" or "error"
    message: str           # human readable result


# ── /run-code ─────────────────────────────────────────────────────────────────

class CodeRequest(BaseModel):
    code:     str
    language: Optional[str] = ""   # hint — "python", "sql", etc.

class CodeResponse(BaseModel):
    language: str
    output:   str          # stdout / execution result
    error:    Optional[str] = None


# ── /notes ────────────────────────────────────────────────────────────────────

class NotesResponse(BaseModel):
    user_id: str
    notes:   dict          # { "topic": "note text", ... }


# ── /profile ──────────────────────────────────────────────────────────────────

class ProfileResponse(BaseModel):
    user_id:  str
    profile:  dict         # { name, level, session_count }
    progress: dict         # { completed, weak, scores }
