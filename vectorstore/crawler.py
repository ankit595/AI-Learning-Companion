# =============================================================================
# vectorstore/crawler.py — lazy on-demand sublink crawler
#
# When the retriever finds low-confidence results for a URL-sourced page,
# this module:
#   1. Extracts all same-domain links from the original page (cached)
#   2. Picks the most relevant un-crawled sublink for the question
#   3. Fetches + ingests it into Chroma immediately
#   4. Marks it crawled (via Chroma metadata) so it's never re-fetched
#
# Limits:
#   MAX_SUBLINKS_PER_DOMAIN = 10   (never crawls more than 10 subpages)
#   FETCH_TIMEOUT           = 10s
#   Same domain only — never follows external links
# =============================================================================

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import time
from urllib.parse import urlparse, urljoin

import config

MAX_SUBLINKS_PER_DOMAIN = int(os.getenv("MAX_CRAWL_SUBLINKS", "10"))
FETCH_TIMEOUT           = 10   # seconds per page fetch
MIN_CONFIDENCE          = float(os.getenv("CRAWL_CONFIDENCE_THRESHOLD", "0.35"))


# ---------------------------------------------------------------------------
# Extract all same-domain hrefs from a URL (best-effort, no JS rendering)
# ---------------------------------------------------------------------------
def extract_sublinks(page_url: str) -> list[str]:
    """
    Fetch page_url and return all unique same-domain absolute hrefs.
    Returns [] on any error.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        parsed  = urlparse(page_url)
        base    = f"{parsed.scheme}://{parsed.netloc}"
        resp    = requests.get(page_url, timeout=FETCH_TIMEOUT,
                               headers={"User-Agent": "AI-Learning-Companion/1.0"})
        resp.raise_for_status()
        soup    = BeautifulSoup(resp.text, "html.parser")
        links   = set()
        for tag in soup.find_all("a", href=True):
            href = tag["href"].split("#")[0].strip()   # strip anchors
            if not href or href.startswith("mailto:") or href.startswith("javascript:"):
                continue
            abs_url = urljoin(base, href)
            # Same domain only
            if urlparse(abs_url).netloc == parsed.netloc and abs_url != page_url:
                links.add(abs_url.rstrip("/"))
        return sorted(links)
    except Exception as e:
        print(f"[crawler] extract_sublinks failed for {page_url}: {e}")
        return []


# ---------------------------------------------------------------------------
# Get already-crawled sublinks for a parent URL from Chroma metadata
# ---------------------------------------------------------------------------
def get_crawled_sublinks(user_id: str, parent_url: str) -> set[str]:
    """
    Returns set of sublink URLs already ingested (crawled=True in metadata).
    """
    try:
        from vectorstore.retriever import get_vectorstore
        vs  = get_vectorstore()
        raw = vs.get(
            where={"$and": [
                {"user_id":    {"$eq": user_id}},
                {"parent_url": {"$eq": parent_url}},
                {"crawled":    {"$eq": "true"}},
            ]},
            include=["metadatas"],
        )
        return {
            m.get("source", "") for m in (raw.get("metadatas") or [])
        }
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Pick the most relevant un-crawled sublink for a question (keyword heuristic)
# ---------------------------------------------------------------------------
def pick_best_sublink(question: str, candidates: list[str]) -> str | None:
    """
    Score each candidate URL by how many question words appear in the URL path.
    Returns the highest-scoring candidate, or the first candidate if no match.
    """
    if not candidates:
        return None

    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    best, best_score = candidates[0], -1

    for url in candidates:
        path  = urlparse(url).path.lower()
        score = sum(1 for w in words if w in path and len(w) > 2)
        if score > best_score:
            best, best_score = url, score

    return best


# ---------------------------------------------------------------------------
# Core: lazy crawl — fetch + ingest ONE relevant sublink if confidence is low
#
# Returns True if a new sublink was ingested, False otherwise.
# ---------------------------------------------------------------------------
def lazy_crawl(
    question:   str,
    user_id:    str,
    parent_url: str,
    topic:      str = "",
) -> bool:
    """
    Called by explainer when retrieval confidence is low.

    1. Extracts all same-domain sublinks from parent_url
    2. Removes already-crawled ones
    3. Picks the most relevant one for `question`
    4. Ingests it with metadata {crawled: "true", parent_url: parent_url}
    5. Returns True so explainer knows to re-run retrieval
    """
    print(f"[crawler] Lazy crawl triggered for: {parent_url}")

    # Check how many sublinks already crawled for this domain
    already_crawled = get_crawled_sublinks(user_id, parent_url)
    if len(already_crawled) >= MAX_SUBLINKS_PER_DOMAIN:
        print(f"[crawler] Max sublinks ({MAX_SUBLINKS_PER_DOMAIN}) reached for {parent_url}")
        return False

    # Extract all sublinks from parent page
    all_links = extract_sublinks(parent_url)
    if not all_links:
        print(f"[crawler] No sublinks found on {parent_url}")
        return False

    # Filter out already-crawled ones
    candidates = [l for l in all_links if l not in already_crawled]
    if not candidates:
        print(f"[crawler] All sublinks already crawled for {parent_url}")
        return False

    # Pick most relevant one
    chosen = pick_best_sublink(question, candidates)
    if not chosen:
        return False

    print(f"[crawler] Fetching sublink: {chosen}")

    # Ingest it — reuse ingest pipeline with extra metadata
    try:
        from vectorstore.ingest import ingest_source
        ingest_source(
            source   = chosen,
            user_id  = user_id,
            topic    = topic or parent_url,
            extra_metadata = {
                "crawled":    "true",
                "parent_url": parent_url,
            },
        )
        print(f"[crawler] ✅ Ingested sublink: {chosen}")
        return True
    except Exception as e:
        print(f"[crawler] ❌ Failed to ingest {chosen}: {e}")
        return False


# ---------------------------------------------------------------------------
# Confidence scorer — estimates how relevant retrieved docs are to question
# Returns a float 0.0–1.0 (higher = more relevant)
# ---------------------------------------------------------------------------
def retrieval_confidence(docs: list, question: str) -> float:
    """
    Simple keyword-overlap heuristic — no extra LLM call needed.
    Returns fraction of question words found across retrieved docs.
    """
    if not docs:
        return 0.0

    words   = set(re.findall(r"[a-z0-9]+", question.lower()))
    content = " ".join(d.page_content.lower() for d in docs)
    if not words:
        return 1.0

    matched = sum(1 for w in words if w in content and len(w) > 2)
    return matched / len([w for w in words if len(w) > 2]) if words else 0.0
