# =============================================================================
# config.py — single source of truth for all paths, model names, constants
# Import this at the top of every file → SSL fix applied everywhere
#
# TO SWITCH PROVIDER: change LLM_PROVIDER below to one of:
#   "openai"   → Direct OpenAI API
#   "groq"     → Groq (fast inference, free tier, llama/mixtral models)
#   "gemini"   → Google Gemini via langchain-google-genai
#   "ollama"   → Local Ollama (no API key, fully offline)
# =============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# ★ PROVIDER SWITCH — change this one line to switch LLM provider
# -----------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")   # override via .env too
print(f"LLM_PROVIDER={LLM_PROVIDER} (from .env or default)")

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR       = Path(__file__).parent
DATA_DIR       = BASE_DIR / "data"
NOTES_DIR      = BASE_DIR / "notes"
CHROMA_DIR     = BASE_DIR / "chroma_db"
SQLITE_PATH    = str(BASE_DIR / "memory.db")
EMBED_CACHE    = str(BASE_DIR / ".embedding_cache")
HF_CACHE       = str(BASE_DIR / ".hf_cache")

# Create dirs if they don't exist
DATA_DIR.mkdir(exist_ok=True)
NOTES_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Provider-specific settings
# llm_factory.py reads these — agents never touch them directly
# -----------------------------------------------------------------------------

# ── Direct OpenAI ─────────────────────────────────────────────────────────────
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
OPENAI_LLM_MODEL  = os.getenv("OPENAI_LLM_MODEL",  "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# ── Groq (free tier — fast llama/mixtral) ────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
GROQ_LLM_MODEL    = os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant")   # fast + cheap; use llama-3.3-70b-versatile for quality

# ── Google Gemini ─────────────────────────────────────────────────────────────
GEMINI_API_KEY    = os.getenv("GOOGLE_API_KEY")
GEMINI_LLM_MODEL  = os.getenv("GEMINI_LLM_MODEL",  "gemini-2.0-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001")

# ── Ollama (local, no API key) ────────────────────────────────────────────────
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
OLLAMA_LLM_MODEL  = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# -----------------------------------------------------------------------------
# Shared convenience aliases (used by llm_factory.py)
# These auto-resolve based on LLM_PROVIDER — don't set manually
# -----------------------------------------------------------------------------
_MODEL_MAP = {
    "openai": OPENAI_LLM_MODEL,
    "groq":   GROQ_LLM_MODEL,
    "gemini": GEMINI_LLM_MODEL,
    "ollama": OLLAMA_LLM_MODEL,
}
LLM_MODEL = _MODEL_MAP.get(LLM_PROVIDER, OPENAI_LLM_MODEL)

# Embeddings — Groq and Ollama(llama) don't offer embeddings API
# so they fall back to OpenAI embeddings (or HuggingFace if no key)
EMBED_VIA_OPENAI = LLM_PROVIDER in ("openai", "groq", "ollama")
EMBED_MODEL = {
    "openai": OPENAI_EMBED_MODEL,
    "gemini": GEMINI_EMBED_MODEL,
    "ollama": OLLAMA_EMBED_MODEL,
}.get(LLM_PROVIDER, OPENAI_EMBED_MODEL)

# -----------------------------------------------------------------------------
# ChromaDB
# -----------------------------------------------------------------------------
CHROMA_COLLECTION = "all_docs"

# -----------------------------------------------------------------------------
# Chunking
# -----------------------------------------------------------------------------
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100

# -----------------------------------------------------------------------------
# Retrieval
# -----------------------------------------------------------------------------
RETRIEVER_K       = 3     # chunks returned to LLM (was 5) — reduce context tokens
RETRIEVER_FETCH_K = 15    # for MMR — fetch more, return diverse k

# -----------------------------------------------------------------------------
# Lazy crawler (on-demand sublink fetching)
# -----------------------------------------------------------------------------
MAX_CRAWL_SUBLINKS        = int(os.getenv("MAX_CRAWL_SUBLINKS", "10"))   # max sublinks per domain
CRAWL_CONFIDENCE_THRESHOLD = float(os.getenv("CRAWL_CONFIDENCE_THRESHOLD", "0.35"))  # trigger below this

# -----------------------------------------------------------------------------
# Memory
# -----------------------------------------------------------------------------
MAX_TOKENS_IN_CONTEXT = 2000   # trim_messages threshold (was 4000) — saves ~half on history

# -----------------------------------------------------------------------------
# Neo4j (Phase 5)
# -----------------------------------------------------------------------------
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
