# =============================================================================
# tools/search.py — search tools for the researcher agent
#
# Two tools:
#   1. WikipediaSearch  — fetches article summaries (free, no API key)
#   2. ChromaSearch     — searches user's own ingested docs
#
# Used by: agents/researcher.py
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # SSL fix on import


# ---------------------------------------------------------------------------
# Tool 1: Wikipedia Search
# ---------------------------------------------------------------------------
def wikipedia_search(query: str, sentences: int = 5) -> dict:
    """
    Search Wikipedia for a query. Returns title + summary + URL.
    Falls back gracefully if article not found or network fails.
    """
    try:
        import wikipedia
        wikipedia.set_lang("en")

        # Search for best matching page title
        results = wikipedia.search(query, results=3)
        if not results:
            return {"source": "wikipedia", "found": False, "content": f"No Wikipedia article found for '{query}'."}

        # Try the top result, fall back to next if disambiguation error
        for title in results:
            try:
                page    = wikipedia.page(title, auto_suggest=False)
                summary = wikipedia.summary(title, sentences=sentences, auto_suggest=False)
                return {
                    "source":  "wikipedia",
                    "found":   True,
                    "title":   page.title,
                    "url":     page.url,
                    "content": summary,
                }
            except wikipedia.exceptions.DisambiguationError as e:
                # Try first disambiguation option
                try:
                    page    = wikipedia.page(e.options[0], auto_suggest=False)
                    summary = wikipedia.summary(e.options[0], sentences=sentences, auto_suggest=False)
                    return {
                        "source":  "wikipedia",
                        "found":   True,
                        "title":   page.title,
                        "url":     page.url,
                        "content": summary,
                    }
                except Exception:
                    continue
            except wikipedia.exceptions.PageError:
                continue

        return {"source": "wikipedia", "found": False, "content": f"Could not load Wikipedia page for '{query}'."}

    except ImportError:
        return {"source": "wikipedia", "found": False, "content": "Wikipedia package not installed. Run: pip install wikipedia"}
    except Exception as e:
        return {"source": "wikipedia", "found": False, "content": f"Wikipedia search failed: {e}"}


# ---------------------------------------------------------------------------
# Tool 2: ChromaDB Search (user's own ingested docs)
# ---------------------------------------------------------------------------
def chroma_search(query: str, user_id: str, k: int = 3) -> dict:
    """
    Search user's ingested documents in ChromaDB.
    Returns list of relevant chunks with citations.
    """
    try:
        from vectorstore.retriever import get_retriever
        retriever = get_retriever(user_id=user_id)

        # Override k for research (want fewer, more focused results)
        retriever.search_kwargs["k"] = k

        docs = retriever.invoke(query)
        if not docs:
            return {"source": "chroma", "found": False, "content": "No relevant documents found in your knowledge base.", "chunks": []}

        chunks = []
        for doc in docs:
            src   = os.path.basename(str(doc.metadata.get("source", "unknown")))
            topic = doc.metadata.get("topic", "")
            chunks.append({
                "citation": f"[{src} | {topic}]",
                "content":  doc.page_content,
            })

        # Combine into one string for easy use in prompts
        combined = "\n\n".join(f"{c['citation']}\n{c['content']}" for c in chunks)
        return {
            "source":  "chroma",
            "found":   True,
            "content": combined,
            "chunks":  chunks,
        }

    except Exception as e:
        return {"source": "chroma", "found": False, "content": f"ChromaDB search failed: {e}", "chunks": []}
