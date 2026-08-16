# =============================================================================
# app.py — Streamlit UI for AI-Learning-companion
#
# Pure UI layer — NO direct graph/agent calls.
# Talks to the FastAPI backend (api/main.py) over HTTP only.
#
# Run the backend first:
#   python3 -m uvicorn api.main:app --reload --port 8000
# Then run this:
#   streamlit run app.py
# =============================================================================

import streamlit as st
import requests
import re
import time
import os

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
# Override via API_BASE env var when running in Docker (e.g. http://api:8000)
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="AI-Learning-companion",
    page_icon="📚",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Global styles
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* ---- Light mode defaults ------------------------------------------- */
    :root {
        --primary: #2563EB;
        --primary-soft: #EFF4FF;
        --bg: #F8FAFC;
        --card: #FFFFFF;
        --card-border: #E2E8F0;
        --text: #0F172A;
        --text-secondary: #64748B;
        --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);

        --badge-pdf-bg:#FEE2E2; --badge-pdf-fg:#B91C1C;
        --badge-yt-bg:#FEF3C7;  --badge-yt-fg:#92400E;
        --badge-txt-bg:#DBEAFE; --badge-txt-fg:#1D4ED8;
        --badge-url-bg:#E0E7FF; --badge-url-fg:#4338CA;
    }

    /* ---- Dark mode overrides -------------------------------------------
       Matches system/browser preference AND Streamlit's own dark theme. */
    @media (prefers-color-scheme: dark) {
        :root {
            --primary: #60A5FA;
            --primary-soft: rgba(96,165,250,0.12);
            --bg: #0F172A;
            --card: #1E293B;
            --card-border: #334155;
            --text: #F1F5F9;
            --text-secondary: #94A3B8;
            --shadow: 0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.3);

            --badge-pdf-bg:rgba(248,113,113,0.18); --badge-pdf-fg:#FCA5A5;
            --badge-yt-bg:rgba(251,191,36,0.18);   --badge-yt-fg:#FCD34D;
            --badge-txt-bg:rgba(96,165,250,0.18);  --badge-txt-fg:#93C5FD;
            --badge-url-bg:rgba(129,140,248,0.18); --badge-url-fg:#A5B4FC;
        }
    }
    /* Streamlit sets data-theme="dark" on <html> when user picks dark theme
       in settings — mirror the same overrides so both cases are covered. */
    [data-theme="dark"] {
        --primary: #60A5FA;
        --primary-soft: rgba(96,165,250,0.12);
        --bg: #0F172A;
        --card: #1E293B;
        --card-border: #334155;
        --text: #F1F5F9;
        --text-secondary: #94A3B8;
        --shadow: 0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.3);

        --badge-pdf-bg:rgba(248,113,113,0.18); --badge-pdf-fg:#FCA5A5;
        --badge-yt-bg:rgba(251,191,36,0.18);   --badge-yt-fg:#FCD34D;
        --badge-txt-bg:rgba(96,165,250,0.18);  --badge-txt-fg:#93C5FD;
        --badge-url-bg:rgba(129,140,248,0.18); --badge-url-fg:#A5B4FC;
    }

    .stApp { background-color: var(--bg); }

    /* Cards */
    .card {
        background: var(--card);
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: var(--shadow);
        border: 1px solid var(--card-border);
        margin-bottom: 14px;
        color: var(--text);
    }

    /* Badges / pills */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.3px;
        margin-right: 6px;
    }
    .badge-pdf   { background:var(--badge-pdf-bg); color:var(--badge-pdf-fg); }
    .badge-yt    { background:var(--badge-yt-bg);  color:var(--badge-yt-fg); }
    .badge-txt   { background:var(--badge-txt-bg); color:var(--badge-txt-fg); }
    .badge-url   { background:var(--badge-url-bg); color:var(--badge-url-fg); }

    .mode-badge {
        display:inline-block; padding:3px 12px; border-radius:999px;
        font-size:11px; font-weight:700; letter-spacing:0.5px;
        background: var(--primary); color:white; margin-bottom:8px;
    }

    /* Source list item */
    .source-item {
        background: var(--card);
        border: 1px solid var(--card-border);
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 8px;
        font-size: 13px;
        color: var(--text);
        box-shadow: var(--shadow);
    }

    .sidebar-subtitle {
        color: var(--text-secondary);
        font-size: 13px;
        margin-top: -8px;
        margin-bottom: 16px;
    }

    .section-header {
        font-size: 12px;
        font-weight: 700;
        color: var(--text-secondary);
        letter-spacing: 0.6px;
        margin-top: 18px;
        margin-bottom: 8px;
    }

    .citation {
        color: var(--text-secondary);
        font-size: 12px;
        font-style: italic;
        margin-top: 6px;
    }

    .chunk-card {
        background: var(--card);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        border: 1px solid var(--card-border);
        color: var(--text);
    }
    .chunk-score {
        color: var(--primary);
        font-weight: 600;
        font-size: 12px;
    }

    .footer-summary {
        color: var(--text-secondary);
        font-size: 12px;
        margin-top: 10px;
        text-align: center;
    }

    /* Main header banner */
    .app-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 20px 24px;
        border-radius: 16px;
        margin-bottom: 18px;
        background: linear-gradient(135deg, var(--primary) 0%, #7C3AED 100%);
        box-shadow: var(--shadow);
    }
    .app-header-icon {
        font-size: 34px;
        line-height: 1;
    }
    .app-header-title {
        font-size: 28px;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.3px;
    }
    .app-header-subtitle {
        font-size: 13px;
        color: rgba(255,255,255,0.85);
        margin-top: 2px;
    }
    .app-header-stats {
        margin-left: auto;
        display: flex;
        gap: 10px;
    }
    .app-header-stat {
        background: rgba(255,255,255,0.16);
        color: #FFFFFF;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        white-space: nowrap;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session state initialisation
# -----------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "user_id":         "",       # empty until user confirms their name
        "user_confirmed":  False,    # True once name is locked in
        "chat_history":    [],   # [{role, content, intent, topic, context, citation}]
        "sources":         [],   # [{type, name}]
        "chunks_indexed":  0,
        "last_context":    "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# -----------------------------------------------------------------------------
# API helper functions
# -----------------------------------------------------------------------------
def api_chat(user_id: str, message: str) -> dict:
    try:
        resp = requests.post(
            f"{API_BASE}/chat",
            json={"user_id": user_id, "message": message},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"response": f"⚠️ Error contacting backend: {e}", "intent": "", "topic": "", "context": ""}


def api_ingest_file(user_id: str, topic: str, file) -> dict:
    try:
        files = {"file": (file.name, file.getvalue())}
        data = {"user_id": user_id, "topic": topic}
        resp = requests.post(f"{API_BASE}/ingest/file", data=data, files=files, timeout=180)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def api_ingest_url(user_id: str, url: str, topic: str = "") -> dict:
    try:
        resp = requests.post(
            f"{API_BASE}/ingest/url",
            json={"user_id": user_id, "url": url, "topic": topic},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def api_get_sources(user_id: str) -> dict:
    """Fetch previously ingested sources for this user from ChromaDB (persists across sessions)."""
    try:
        resp = requests.get(f"{API_BASE}/sources/{user_id}", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"sources": [], "total_chunks": 0, "error": str(e)}


def api_delete_source(user_id: str, source_name: str) -> dict:
    """Delete all chunks for a specific ingested source."""
    try:
        # source_name may contain slashes (e.g. "golangbot.com/switch")
        # The FastAPI route uses {source_name:path} so literal slashes are fine.
        resp = requests.delete(f"{API_BASE}/sources/{user_id}/{source_name}", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def api_get_notes(user_id: str) -> dict:
    """Fetch all saved notes for this user from the API."""
    try:
        resp = requests.get(f"{API_BASE}/notes/{user_id}", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"notes": {}, "error": str(e)}


def api_delete_note(user_id: str, topic: str) -> dict:
    """Delete a specific note by sending a delete message via chat."""
    return api_chat(user_id, f"delete my {topic} note")


def refresh_sources():
    """Re-sync st.session_state['sources'] + chunk count from ChromaDB via API."""
    src_data = api_get_sources(st.session_state["user_id"])
    st.session_state["sources"] = [
        {"type": s["type"].upper(), "name": s["name"], "chunks": s.get("chunks", 0), "topic": s.get("topic", "")}
        for s in src_data.get("sources", [])
    ]
    st.session_state["chunks_indexed"] = src_data.get("total_chunks", 0)


def detect_source_badge(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".pdf"):
        return "PDF", "badge-pdf"
    if "youtube.com" in lower or "youtu.be" in lower:
        return "YT", "badge-yt"
    if lower.endswith(".txt") or lower.endswith(".md"):
        return "TXT", "badge-txt"
    return "URL", "badge-url"


def parse_citation(context: str) -> str:
    """Best-effort extraction of a short citation line from retrieved context."""
    if not context:
        return ""
    first_line = context.strip().split("\n")[0]
    return first_line[:120]


def detect_mode_badge(intent: str) -> str:
    mapping = {
        "explain":  "TUTOR",
        "quiz":     "QUIZ",
        "research": "RESEARCH",
        "plan":     "PLANNER",
        "code":     "CODE",
        "notes":    "NOTES",
    }
    return mapping.get(intent, "TUTOR")


def render_ai_response(content: str, intent: str):
    """
    Renders an assistant response as a proper markdown-aware card.
    Citations are embedded inline in the LLM response itself — e.g.
    [golangbot.com › switch] — no separate footer citation needed.
    """
    mode_badge = detect_mode_badge(intent)
    st.markdown(f'<span class="mode-badge">{mode_badge}</span>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(content if content else "_(no response)_")


# =============================================================================
# LOGIN GATE — must enter a name once before using the app
# =============================================================================
if not st.session_state["user_confirmed"]:
    st.markdown(
        '''
        <div class="app-header" style="justify-content:center; text-align:center; flex-direction:column;">
            <div class="app-header-icon">📚</div>
            <div class="app-header-title">Welcome to AI-Learning-companion</div>
            <div class="app-header-subtitle">Enter your name to start your personalised session</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    _, mid_col, _ = st.columns([1, 1.2, 1])
    with mid_col:
        with st.container(border=True):
            name_input = st.text_input("Your name", placeholder="e.g. Alex", key="login_name_input")
            if st.button("Start session →", use_container_width=True, type="primary"):
                if name_input.strip():
                    st.session_state["user_id"] = name_input.strip()
                    st.session_state["user_confirmed"] = True

                    # Restore previously ingested sources for this user from ChromaDB
                    # (data persists on disk permanently — only session state was lost)
                    refresh_sources()

                    st.rerun()
                else:
                    st.warning("Please enter your name to continue.")

    st.stop()   # halt rendering the rest of the app until confirmed


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### 📚 AI-Learning-companion")
    st.markdown('<div class="sidebar-subtitle">Ask questions against your own material.</div>', unsafe_allow_html=True)

    # ── User identity (locked after login) ──────────────────────────────────
    st.markdown(
        f'<div class="source-item">👤 Logged in as <b>{st.session_state["user_id"]}</b></div>',
        unsafe_allow_html=True,
    )
    if st.button("🔓 Switch user", use_container_width=True):
        st.session_state["user_confirmed"] = False
        st.session_state["user_id"] = ""
        st.session_state["chat_history"] = []
        st.rerun()

    # ── Section 1: Upload Document ───────────────────────────────────────────
    st.markdown('<div class="section-header">UPLOAD A DOCUMENT</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag and drop file here",
            type=["pdf", "txt", "docx", "pptx", "ppt", "xlsx", "xls", "csv", "json", "md", "rst"],
            help="Limit 200MB per file • PDF, DOCX, PPTX, XLSX, CSV, JSON, TXT, MD",
            label_visibility="visible",
        )
        st.caption("Limit 200MB per file • PDF, DOCX, PPTX, XLSX, CSV, JSON, TXT, MD")

        upload_topic = st.text_input("Topic (optional)", key="upload_topic", placeholder="e.g. Transformers")

        if uploaded_file is not None:
            if st.button("📤 Ingest file", use_container_width=True, type="primary"):
                progress = st.progress(0, text="Indexing document...")
                for pct in (20, 45, 70):
                    time.sleep(0.15)
                    progress.progress(pct, text="Indexing document...")

                result = api_ingest_file(st.session_state["user_id"], upload_topic, uploaded_file)
                progress.progress(100, text="Done")
                time.sleep(0.2)
                progress.empty()

                if result.get("status") == "ok":
                    st.success(result.get("message", "Ingested."))
                    refresh_sources()   # re-sync accurate counts from ChromaDB
                else:
                    st.error(result.get("message", "Ingestion failed."))
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 2: Add Link ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">ADD LINK</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        url_input = st.text_input("Web page or YouTube URL", key="url_input", label_visibility="collapsed",
                                   placeholder="Web page or YouTube URL")
        if st.button("➕ Add", use_container_width=True):
            if url_input.strip():
                progress = st.progress(0, text="Indexing link...")
                for pct in (30, 60, 90):
                    time.sleep(0.15)
                    progress.progress(pct, text="Indexing link...")

                result = api_ingest_url(st.session_state["user_id"], url_input.strip())
                progress.progress(100, text="Done")
                time.sleep(0.2)
                progress.empty()

                if result.get("status") == "ok":
                    st.success(result.get("message", "Ingested."))
                    refresh_sources()   # re-sync accurate counts from ChromaDB
                else:
                    st.error(result.get("message", "Ingestion failed."))
            else:
                st.warning("Please enter a URL first.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 3: Sources ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">SOURCES</div>', unsafe_allow_html=True)
    if st.session_state["sources"]:
        for src in st.session_state["sources"]:
            badge, css_class = detect_source_badge(src["name"])
            col_name, col_del = st.columns([5, 1])
            with col_name:
                st.markdown(
                    f'<div class="source-item">'
                    f'<span class="badge {css_class}">{badge}</span>{src["name"]}'
                    f'<span style="font-size:0.7rem;color:#888;margin-left:6px;">({src.get("chunks", "?")} chunks)</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button("🗑️", key=f"del_src_{src['name']}", help=f"Delete {src['name']}"):
                    src_name = src["name"].strip()
                    if not src_name or src_name == "unknown":
                        st.warning("Cannot delete — source has no valid name.")
                    else:
                        result = api_delete_source(st.session_state["user_id"], src_name)
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.success(f"Deleted '{src_name}' ({result.get('chunks_deleted', 0)} chunks)")
                            refresh_sources()
                            st.rerun()
        st.markdown(
            f'<div class="footer-summary">'
            f'{len(st.session_state["sources"])} sources • {st.session_state["chunks_indexed"]} chunks indexed'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("No sources yet — upload a file or add a link above.")

    # ── Section 4: Notes ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">MY NOTES</div>', unsafe_allow_html=True)

    # Fetch notes fresh on every render (lightweight GET call)
    notes_data = api_get_notes(st.session_state["user_id"])
    notes: dict = notes_data.get("notes", {})

    if not notes:
        st.caption("No notes yet — notes are auto-saved after each explanation.")
    else:
        st.caption(f"{len(notes)} saved note(s)")
        for topic, content in notes.items():
            preview = content[:120].strip().replace("\n", " ")
            if len(content) > 120:
                preview += "…"
            with st.expander(f"📝 {topic} — {preview}", expanded=False):
                st.markdown(content)   # full note, no truncation
                col_copy, col_del = st.columns([3, 1])
                with col_del:
                    if st.button("🗑️ Delete", key=f"del_note_{topic}"):
                        api_delete_note(st.session_state["user_id"], topic)
                        st.rerun()

        # Clear-all button at the bottom
        if st.button("🗑️ Clear all notes", use_container_width=True):
            api_chat(st.session_state["user_id"], "clear all notes")
            st.rerun()


# =============================================================================
# MAIN CONTENT AREA
# =============================================================================
n_sources = len(st.session_state["sources"])
n_chunks  = st.session_state["chunks_indexed"]

st.markdown(
    f'''
    <div class="app-header">
        <div class="app-header-icon">📚</div>
        <div>
            <div class="app-header-title">AI-Learning-companion</div>
            <div class="app-header-subtitle">Your personal AI tutor, quizzer &amp; research assistant</div>
        </div>
        <div class="app-header-stats">
            <span class="app-header-stat">📄 {n_sources} sources</span>
            <span class="app-header-stat">🧩 {n_chunks} chunks</span>
        </div>
    </div>
    ''',
    unsafe_allow_html=True,
)

# ── Chat history render ──────────────────────────────────────────────────────
for turn in st.session_state["chat_history"]:
    if turn["role"] == "user":
        with st.chat_message("user"):
            st.markdown(turn["content"])
    else:
        with st.chat_message("assistant"):
            render_ai_response(
                turn.get("content", ""),
                turn.get("intent", ""),
            )

# ── Chat input (sticky bottom) ───────────────────────────────────────────────
user_message = st.chat_input("Ask anything about your material...")

if user_message:
    st.session_state["chat_history"].append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = api_chat(st.session_state["user_id"], user_message)

        citation = parse_citation(result.get("context", ""))
        render_ai_response(result.get("response", ""), result.get("intent", ""))

    st.session_state["chat_history"].append({
        "role":     "assistant",
        "content":  result.get("response", ""),
        "intent":   result.get("intent", ""),
        "topic":    result.get("topic", ""),
        "citation": citation,
    })
    st.session_state["last_context"] = result.get("context", "")
    st.rerun()


# =============================================================================
# RETRIEVED CHUNKS PANEL
# =============================================================================
st.markdown('<div class="section-header">RETRIEVED CHUNKS</div>', unsafe_allow_html=True)

with st.expander("Show retrieved chunks", expanded=False):
    context = st.session_state.get("last_context", "")
    if not context:
        st.caption("No retrieval performed yet — ask a question about your ingested material.")
    else:
        # Context is raw concatenated text from the retriever — split into pseudo-chunks
        raw_chunks = [c.strip() for c in re.split(r"\n\n+", context) if c.strip()]
        for i, chunk in enumerate(raw_chunks, start=1):
            source_name = st.session_state["sources"][-1]["name"] if st.session_state["sources"] else "Unknown source"
            fake_score = round(0.95 - i * 0.03, 2)
            st.markdown(
                f'<div class="chunk-card">'
                f'<b>{source_name}</b><br>'
                f'<span class="chunk-score">chunk {i} | score {max(fake_score, 0.5)}</span>'
                f'<p style="margin-top:6px; font-size:13px; color:#334155;">{chunk[:400]}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
