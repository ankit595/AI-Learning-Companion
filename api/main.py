# =============================================================================
# api/main.py — FastAPI application entry point
#
# Mounts all routers. Run with:
#   uvicorn api.main:app --reload --port 8000
#
# Auto docs available at:
#   http://localhost:8000/docs
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config   # loads .env + SSL fix for netapp provider

from fastapi import FastAPI
from api.routers import chat, ingest, notes, profile, code, sources

app = FastAPI(
    title       = "AI Learning Companion API",
    description = "Backend API for the multi-agent AI Learning Companion",
    version     = "1.0.0",
)

# ── Mount all routers ──────────────────────────────────────────────────────────
app.include_router(chat.router)
app.include_router(ingest.router)
app.include_router(notes.router)
app.include_router(profile.router)
app.include_router(code.router)
app.include_router(sources.router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "service": "AI Learning Companion API"}


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", tags=["health"])
def root():
    return {
        "message": "AI Learning Companion API",
        "docs":    "http://localhost:8000/docs",
        "endpoints": [
            "POST /chat",
            "POST /ingest/file",
            "POST /ingest/url",
            "GET  /notes/{user_id}",
            "GET  /profile/{user_id}",
            "POST /run-code",
        ]
    }
