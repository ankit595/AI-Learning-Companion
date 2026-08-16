# =============================================================================
#    # ── BM25 retriever over the same filtered subset ────────────────────────
    # BM25Retriever has no native metadata filter — pull matching docs out of
    # Chroma (same filter) and index them in-memory.
    # Falls back to vector-only if nothing is indexed or construction fails.orstore/retriever.py — load existing ChromaDB and return a retriever
#
# Usage (in agents):
#   from vectorstore.retriever import get_retriever
#   retriever = get_retriever(topic="Python")   # filtered by topic
#   retriever = get_retriever()                 # all docs
#
#   docs = retriever.invoke("what is attention mechanism?")
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # SSL fix applied on import
from llm_factory import get_embedder as _factory_embedder

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever


def get_embedder():
    """Delegates to llm_factory — provider-agnostic embedder."""
    return _factory_embedder()


def get_vectorstore() -> Chroma:
    """Open the persisted ChromaDB collection."""
    embedder = get_embedder()
    return Chroma(
        persist_directory=str(config.CHROMA_DIR),
        embedding_function=embedder,
        collection_name=config.CHROMA_COLLECTION,
    )


# -----------------------------------------------------------------------------
# Retriever — Hybrid: BM25 (keyword) + Chroma MMR (vector), merged 50/50.
# BM25 catches exact term matches (function names, error codes, acronyms)
# that vector search can blur; Chroma MMR catches conceptual/semantic matches.
# Optional topic filter — only search chunks with matching metadata.
# -----------------------------------------------------------------------------
def get_retriever(topic: str = None, user_id: str = None):
    """
    Hybrid retriever: BM25 (keyword) + Chroma MMR (semantic), merged 50/50.
    Optionally scoped to a specific user_id and/or topic via Chroma metadata filters.
    Falls back to vector-only if BM25 construction fails or no docs exist.
    """
    vectorstore = get_vectorstore()

    search_kwargs = {
        "k":       config.RETRIEVER_K,
        "fetch_k": config.RETRIEVER_FETCH_K,
    }

    # Build filter — always scope by user_id, optionally also by topic.
    # Chroma requires "$and" when combining more than one key in a `where`
    # clause — a single key can be passed as-is.
    conditions = []
    if user_id:
        conditions.append({"user_id": user_id})
    if topic:
        conditions.append({"topic": topic})

    filters = None
    if len(conditions) == 1:
        filters = conditions[0]
    elif len(conditions) > 1:
        filters = {"$and": conditions}

    if filters:
        search_kwargs["filter"] = filters

    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )

    # ── Build a BM25 retriever over the same filtered subset ────────────────
    # BM25Retriever has no native metadata filter, so we pull the matching
    # documents straight out of Chroma (same filter as above) and index them
    # in-memory. Falls back to vector-only if there's nothing to index yet
    # (e.g. no docs ingested, or BM25 construction fails for any reason).
    try:
        raw = vectorstore.get(
            where=filters if filters else None,
            include=["documents", "metadatas"],
        )
        texts     = raw.get("documents", []) or []
        metadatas = raw.get("metadatas", []) or []

        if not texts:
            return vector_retriever

        bm25_docs = [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(texts, metadatas)
        ]
        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = config.RETRIEVER_K

        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5],
        )
    except Exception as e:
        print(f"[retriever] BM25 hybrid setup failed, falling back to vector-only: {e}")
        return vector_retriever


def _resolve_source_name(raw_source: str) -> str:
    """
    Canonical display name for a source — used by both list and delete
    so they always agree on the same string.
      URLs  → "golangbot.com/switch"   (netloc + stripped path)
      Files → "notes.pdf"             (basename)
      Empty → "unknown"
    """
    src = str(raw_source).strip().rstrip("/")
    if not src:
        return "unknown"
    if src.startswith("http://") or src.startswith("https://"):
        from urllib.parse import urlparse
        parsed = urlparse(src)
        # Use netloc + path as the stable key — avoids empty basename on root URLs
        path = parsed.path.strip("/")
        return f"{parsed.netloc}/{path}" if path else parsed.netloc
    return os.path.basename(src) or src or "unknown"


def list_user_sources(user_id: str) -> list:
    """Return distinct ingested sources for a user with chunk counts.
    Reads directly from ChromaDB metadata — persists across restarts.
    Returns: [{"name", "type", "topic", "chunks"}, ...]
    """
    vectorstore = get_vectorstore()
    raw = vectorstore.get(where={"user_id": user_id}, include=["metadatas"])
    metadatas = raw.get("metadatas", []) or []

    sources = {}
    for meta in metadatas:
        source    = _resolve_source_name(meta.get("source", ""))
        file_type = meta.get("file_type", "txt")
        topic     = meta.get("topic", "")
        if source not in sources:
            sources[source] = {"name": source, "type": file_type, "topic": topic, "chunks": 0}
        sources[source]["chunks"] += 1

    return list(sources.values())


def count_user_chunks(user_id: str) -> int:
    """Total number of chunks ingested for this user across all sources."""
    vectorstore = get_vectorstore()
    raw = vectorstore.get(where={"user_id": user_id}, include=[])
    return len(raw.get("ids", []) or [])


def delete_user_source(user_id: str, source_name: str) -> int:
    """Delete all chunks for a source. Matches via _resolve_source_name()
    so the name always aligns with what list_user_sources() returns.
    Returns number of chunks deleted.
    """
    vectorstore = get_vectorstore()
    raw = vectorstore.get(where={"user_id": user_id}, include=["metadatas"])
    ids       = raw.get("ids", []) or []
    metadatas = raw.get("metadatas", []) or []

    to_delete = [
        doc_id for doc_id, meta in zip(ids, metadatas)
        if _resolve_source_name(meta.get("source", "")) == source_name
    ]

    if to_delete:
        vectorstore.delete(ids=to_delete)

    return len(to_delete)


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "What is this document about?"
    print(f"[retriever] Query: {query}\n")

    retriever = get_retriever()
    docs = retriever.invoke(query)

    if not docs:
        print("[retriever] No results. Has anything been ingested yet?")
    else:
        for i, doc in enumerate(docs, 1):
            src   = doc.metadata.get("source", "?")
            topic = doc.metadata.get("topic",  "?")
            page  = doc.metadata.get("page",   "")
            page_str = f" p.{page}" if page != "" else ""
            print(f"--- Result {i} | topic={topic} | {src}{page_str} ---")
            print(doc.page_content[:300])
            print()
