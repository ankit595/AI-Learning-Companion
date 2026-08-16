# =============================================================================
# api/routers/code.py — POST /run-code
#
# Receives { code, language } → safety check → run → return output.
# This is the isolated code execution endpoint.
# In production this router lives in its OWN Docker container.
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException
from tools.code_runner import run_code, safety_check, extract_code, detect_language
from api.models import CodeRequest, CodeResponse

router = APIRouter(prefix="/run-code", tags=["code runner"])


@router.post("", response_model=CodeResponse)
def run_user_code(request: CodeRequest):
    """
    Execute user-submitted code safely.
    - Extracts code from markdown blocks if present
    - Detects language
    - Safety checks for dangerous patterns
    - Runs in subprocess with timeout
    """
    raw = request.code.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="No code provided")

    # Extract from ```python ... ``` blocks if wrapped
    code = extract_code(raw) or raw
    language = detect_language(code, hint=request.language)

    # Safety check before running
    safe, reason = safety_check(code)
    if not safe:
        raise HTTPException(
            status_code=400,
            detail=f"Code blocked — unsafe pattern detected: {reason}"
        )

    result = run_code(code, hint=request.language)

    return CodeResponse(
        language = language,
        output   = result.get("output", ""),
        error    = result.get("error"),
    )
