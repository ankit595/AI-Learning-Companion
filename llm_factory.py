# =============================================================================
# llm_factory.py — single place to get LLM + Embedder for any provider
#
# All agents call:
#   from llm_factory import get_llm, get_embedder
#
# To switch provider: change LLM_PROVIDER in config.py (or .env)
# No changes needed in any agent file.
#
# Supported providers:
#   netapp  → OpenAI-compatible NetApp proxy (current)
#   openai  → Direct OpenAI API
#   groq    → Groq fast inference (free tier)
#   gemini  → Google Gemini
#   ollama  → Local Ollama (fully offline)
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


# ---------------------------------------------------------------------------
# get_llm(temperature) — returns the correct chat LLM for current provider
# ---------------------------------------------------------------------------
def get_llm(temperature: float = 0.3):
    provider = config.LLM_PROVIDER

    # ── NetApp proxy (OpenAI-compatible) ─────────────────────────────────────
    if provider == "netapp":
        import httpx
        from langchain_openai import ChatOpenAI
        # NetApp uses a corporate CA — disable SSL verification for the proxy.
        # This is safe: we're on corp VPN talking to an internal endpoint.
        _http_client = httpx.Client(verify=False)
        return ChatOpenAI(
            model=config.NETAPP_LLM_MODEL,
            base_url=config.NETAPP_BASE_URL,
            api_key=config.NETAPP_API_KEY,
            model_kwargs={"user": config.NETAPP_USER},
            temperature=temperature,
            http_client=_http_client,
        )

    # ── Direct OpenAI ─────────────────────────────────────────────────────────
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.OPENAI_LLM_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=temperature,
        )

    # ── Groq ──────────────────────────────────────────────────────────────────
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=config.GROQ_LLM_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=temperature,
        )

    # ── Google Gemini ─────────────────────────────────────────────────────────
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_LLM_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=temperature,
        )

    # ── Ollama (local) ────────────────────────────────────────────────────────
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.OLLAMA_LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: '{provider}'. "
        "Choose: netapp | openai | groq | gemini | ollama"
    )


# ---------------------------------------------------------------------------
# get_embedder() — returns the correct embedder for current provider
#
# Note: Groq has no embeddings API → falls back to OpenAI embeddings
#       Ollama uses nomic-embed-text (local, no API key needed)
# ---------------------------------------------------------------------------
def get_embedder():
    provider = config.LLM_PROVIDER

    # ── NetApp proxy ──────────────────────────────────────────────────────────
    if provider == "netapp":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.NETAPP_EMBED_MODEL,
            base_url=config.NETAPP_BASE_URL,
            api_key=config.NETAPP_API_KEY,
            model_kwargs={"user": config.NETAPP_USER},
        )

    # ── Direct OpenAI ─────────────────────────────────────────────────────────
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBED_MODEL,
            api_key=config.OPENAI_API_KEY,
        )

    # ── Groq — no embeddings API, fall back to OpenAI ─────────────────────────
    if provider == "groq":
        if config.OPENAI_API_KEY:
            from langchain_openai import OpenAIEmbeddings
            
            return OpenAIEmbeddings(
                model=config.OPENAI_EMBED_MODEL,
                api_key=config.OPENAI_API_KEY,
            )
        # No OpenAI key either → use local HuggingFace
        return _hf_embedder()

    # ── Google Gemini ─────────────────────────────────────────────────────────
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=config.GEMINI_EMBED_MODEL,
            google_api_key=config.GEMINI_API_KEY,
        )

    # ── Ollama (local) ────────────────────────────────────────────────────────
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=config.OLLAMA_EMBED_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: '{provider}'. "
        "Choose: netapp | openai | groq | gemini | ollama"
    )


# ---------------------------------------------------------------------------
# HuggingFace local embedder — universal fallback (no API key needed)
# ---------------------------------------------------------------------------
def _hf_embedder():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        cache_folder=config.HF_CACHE,
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------------------------
# get_llm_with_structured_output(schema, temperature) — for supervisor/quizzer
# ---------------------------------------------------------------------------
def get_llm_with_structured_output(schema, temperature: float = 0):
    return get_llm(temperature=temperature).with_structured_output(schema)