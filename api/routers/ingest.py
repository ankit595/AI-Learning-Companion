# =============================================================================
# api/routers/ingest.py — POST /ingest/file and POST /ingest/url
#
# File upload  → save to temp → call ingest(path, topic, user_id)
# URL ingest   → call ingest(url, topic, user_id) directly
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from vectorstore.ingest import ingest
from api.models import IngestURLRequest, IngestResponse

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/file", response_model=IngestResponse)
async def ingest_file(
    user_id: str  = Form(...),    # form field (not JSON — multipart)
    topic:   str  = Form(""),     # optional topic hint
    file:    UploadFile = File(...),
):
    """
    Upload a PDF or text file to ingest into ChromaDB.
    Streamlit calls this with st.file_uploader output.
    """
    try:
        contents = await file.read()

        # Save to temp file — ingest() needs a file path
        suffix = os.path.splitext(file.filename)[-1] or ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        ingest(source=tmp_path, topic=topic or file.filename, user_id=user_id)
        os.unlink(tmp_path)   # clean up temp file

        return IngestResponse(
            status  = "ok",
            message = f"Ingested '{file.filename}' for user '{user_id}'"
        )

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{str(e)}\n\n{tb}")


@router.post("/url", response_model=IngestResponse)
async def ingest_url(request: IngestURLRequest):
    """
    Ingest a URL or Wikipedia article into ChromaDB.
    """
    try:
        ingest(
            source  = request.url,
            topic   = request.topic or request.url,
            user_id = request.user_id,
        )
        return IngestResponse(
            status  = "ok",
            message = f"Ingested '{request.url}' for user '{request.user_id}'"
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{str(e)}\n\n{tb}")
