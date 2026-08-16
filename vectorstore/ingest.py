# =============================================================================
# vectorstore/ingest.py — load any file → chunk → embed → store in ChromaDB
#
# Metadata AUTO-DETECTED from source:
#   source      → filename or URL
#   file_type   → pdf / url / txt / csv / json / youtube / directory
#   topic       → passed by user OR auto-derived from filename
#   page        → page number (PDFs, from loader)
#   ingested_at → ISO timestamp
#   language    → auto-detected via langdetect
#
# Usage:
#   python -m vectorstore.ingest --source data/myfile.pdf --topic "Python"
#   python -m vectorstore.ingest --source https://youtube.com/watch?v=xxx
#   python -m vectorstore.ingest --source data/notes.txt
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import re
from pathlib import Path
from datetime import datetime

import config  # SSL fix applied on import
from llm_factory import get_llm, get_embedder as _factory_embedder

from langchain_community.document_loaders import (
    PyPDFLoader, WebBaseLoader, TextLoader,
    CSVLoader, JSONLoader, DirectoryLoader, YoutubeLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# ---------------------------------------------------------------------------
# Embedder — delegates to llm_factory (provider-agnostic)
# ---------------------------------------------------------------------------
def get_embedder():
    return _factory_embedder()



# ---------------------------------------------------------------------------
# Auto-detect language
# ---------------------------------------------------------------------------
def detect_language(text: str) -> str:
    try:
        from langdetect import detect
        return detect(text[:500])
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Loader dispatcher → returns (docs, file_type)
# ---------------------------------------------------------------------------
def load_source(source: str) -> tuple:
    source = source.strip()

    if "youtube.com" in source or "youtu.be" in source:
        print(f"[ingest] YouTube: {source}")
        return YoutubeLoader.from_youtube_url(
            source, add_video_info=False, language=["en"]
        ).load(), "youtube"

    if source.startswith("http://") or source.startswith("https://"):
        print(f"[ingest] URL: {source}")
        return WebBaseLoader(source).load(), "url"

    path = Path(source)

    if path.is_dir():
        print(f"[ingest] Directory: {source}")
        return DirectoryLoader(source, glob="**/*.*", show_progress=True).load(), "directory"

    ext = path.suffix.lower()

    if ext == ".pdf":
        print(f"[ingest] PDF: {source}")
        return PyPDFLoader(source).load(), "pdf"

    elif ext == ".csv":
        print(f"[ingest] CSV: {source}")
        return CSVLoader(file_path=source).load(), "csv"

    elif ext == ".json":
        print(f"[ingest] JSON: {source}")
        return JSONLoader(file_path=source, jq_schema=".[]", text_content=False).load(), "json"

    elif ext == ".docx":
        print(f"[ingest] DOCX: {source}")
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(source).load(), "docx"

    elif ext in (".pptx", ".ppt"):
        print(f"[ingest] PPTX: {source}")
        from langchain_community.document_loaders import UnstructuredPowerPointLoader
        return UnstructuredPowerPointLoader(source).load(), "pptx"

    elif ext in (".xlsx", ".xls"):
        print(f"[ingest] Excel: {source}")
        from langchain_community.document_loaders import UnstructuredExcelLoader
        return UnstructuredExcelLoader(source, mode="elements").load(), "xlsx"

    else:  # .txt .md .rst and unknown
        print(f"[ingest] Text ({ext}): {source}")
        return TextLoader(source, encoding="utf-8").load(), "txt"


# ---------------------------------------------------------------------------
# Auto-derive topic — first tries LLM on first page, falls back to filename
# ---------------------------------------------------------------------------
def derive_topic(source: str, first_page_text: str = None) -> str:
    """
    If first_page_text is provided, ask the LLM to identify the topic.
    Falls back to filename/URL parsing if LLM call fails.
    """
    if first_page_text:
        try:
            from llm_factory import get_llm
            llm = get_llm(temperature=0)
            prompt = (
                "Read the following text (first page of a document) and reply with "
                "ONLY a short topic label (2-5 words, no punctuation). "
                "Examples: 'Korean language phrases', 'Docker containers basics', "
                "'LangGraph multi-agent systems'.\n\n"
                f"Text:\n{first_page_text[:1000]}"
            )
            topic = llm.invoke(prompt).content.strip().strip('"').strip("'")
            print(f"[ingest] LLM-detected topic: '{topic}'")
            return topic
        except Exception as e:
            print(f"[ingest] LLM topic detection failed ({e}), falling back to filename")

    # Fallback: derive from filename or URL
    if source.startswith("http"):
        from urllib.parse import urlparse
        return urlparse(source).netloc.replace("www.", "")
    return Path(source).stem.replace("_", " ").replace("-", " ")


# ---------------------------------------------------------------------------
# Clean extracted text
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # CamelCase fix
    text = re.sub(r'\s+', ' ', text)                   # collapse whitespace
    return text.strip()


# ---------------------------------------------------------------------------
# Main ingest function
# ---------------------------------------------------------------------------
def ingest(source: str, topic: str = None, user_id: str = "shared") -> int:
    """
    Load → clean → chunk → auto-tag metadata → store in ChromaDB.
    Returns number of chunks stored.
    user_id scopes docs so each user only retrieves their own data.
    """
    return ingest_source(source, topic=topic, user_id=user_id)


def ingest_source(
    source: str,
    topic: str = None,
    user_id: str = "shared",
    extra_metadata: dict = None,
) -> int:
    """
    Full ingest pipeline with optional extra_metadata (used by lazy crawler
    to stamp crawled=true + parent_url on sublink chunks).
    Returns number of chunks stored.
    """
    docs, file_type = load_source(source)
    if not docs:
        print("[ingest] No documents loaded.")
        return 0
    print(f"[ingest] Loaded {len(docs)} doc(s) | type={file_type}")

    if not topic:
        first_page = docs[0].page_content if docs else ""
        topic = derive_topic(source, first_page_text=first_page)
        print(f"[ingest] Auto-derived topic: '{topic}'")

    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    # Drop pages with no extractable text (e.g. scanned/image-only PDFs)
    docs = [d for d in docs if len(d.page_content.strip()) > 20]
    if not docs:
        raise ValueError(
            "No text could be extracted from this file. "
            "It may be a scanned/image-only PDF. "
            "Please use a PDF with selectable text, or paste the content as a .txt file."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    # Ensure all content is a plain str — HuggingFace tokenizer crashes on None/bytes
    for c in chunks:
        c.page_content = str(c.page_content or "").strip()
    # Drop empty/whitespace-only chunks
    chunks = [c for c in chunks if len(c.page_content) > 10]
    print(f"[ingest] Split into {len(chunks)} chunks")

    ingested_at = datetime.now().isoformat()
    lang = detect_language(chunks[0].page_content if chunks else "")

    for chunk in chunks:
        chunk.metadata["topic"]       = topic
        chunk.metadata["file_type"]   = file_type
        chunk.metadata["source"]      = chunk.metadata.get("source", source)
        chunk.metadata["ingested_at"] = ingested_at
        chunk.metadata["language"]    = lang
        chunk.metadata["user_id"]     = user_id
        # Crawler stamps: crawled="true", parent_url="https://..."
        if extra_metadata:
            chunk.metadata.update(extra_metadata)

    embedder = get_embedder()
    vectorstore = Chroma(
        persist_directory=str(config.CHROMA_DIR),
        embedding_function=embedder,
        collection_name=config.CHROMA_COLLECTION,
    )
    vectorstore.add_documents(chunks)
    print(f"[ingest] ✅ {len(chunks)} chunks | topic='{topic}' | lang={lang} | type={file_type}")
    return len(chunks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path, URL, or YouTube link")
    parser.add_argument("--topic",  default=None,  help="Topic (auto-derived if omitted)")
    args = parser.parse_args()
    ingest(args.source, args.topic)