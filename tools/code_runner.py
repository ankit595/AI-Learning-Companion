# =============================================================================
# tools/code_runner.py — safely runs code snippets in a subprocess sandbox
#
# Currently supported: Python only (JS/Bash planned later)
#
# Safety rules:
#   - Runs in subprocess with hard timeout (5s default)
#   - Blocks dangerous builtins: os.system, open, eval, exec, __import__
#   - Limits output size to 5000 chars
#   - No network access (requests/urllib blocked)
#   - No file system writes (open blocked)
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import subprocess
import tempfile
import time

MAX_EXEC_TIME   = 5       # seconds
MAX_OUTPUT_SIZE = 5000    # chars

# ---------------------------------------------------------------------------
# Blocked patterns — raise error if found in code before execution
# ---------------------------------------------------------------------------
BLOCKED_PATTERNS = [
    (r'\bos\.system\b',       "os.system() not allowed"),
    (r'\bos\.popen\b',        "os.popen() not allowed"),
    (r'\bsubprocess\b',       "subprocess not allowed"),
    (r'\bopen\s*\(',          "file open() not allowed"),
    (r'\b__import__\s*\(',    "__import__() not allowed"),
    (r'\beval\s*\(',          "eval() not allowed"),
    (r'\bexec\s*\(',          "exec() not allowed"),
    (r'\brequests\b',         "network access (requests) not allowed"),
    (r'\burllib\b',           "network access (urllib) not allowed"),
    (r'\bshutil\b',           "shutil not allowed"),
    (r'\bpickle\b',           "pickle not allowed"),
    (r'\bimport\s+socket\b',  "socket not allowed"),
]


# ---------------------------------------------------------------------------
# Language detection — 3 layers
# ---------------------------------------------------------------------------
def detect_language(code: str, hint: str = "") -> str:
    """
    Layer 1: explicit hint from caller (e.g. from ```python tag)
    Layer 2: syntax heuristics
    Layer 3: default to Python
    """
    if hint:
        return hint.lower().strip()

    code_lower = code.lower()

    # Python signals
    py_signals = ["def ", "print(", "import ", "class ", "elif ", "lambda ",
                  "self.", "f\"", "f'", "->", "if __name__"]
    if any(s in code for s in py_signals):
        return "python"

    # JavaScript signals
    js_signals = ["console.log(", "const ", "let ", "var ", "=>", "function(",
                  "document.", "require(", "module.exports"]
    if any(s in code for s in js_signals):
        return "javascript"

    # Bash signals
    bash_signals = ["#!/bin/bash", "echo ", "grep ", "awk ", "sed ", "ls -", "cd "]
    if any(s in code for s in bash_signals):
        return "bash"

    return "python"  # safe default


# ---------------------------------------------------------------------------
# Extract code block from message
# Returns (code, language_hint)
# ---------------------------------------------------------------------------
def extract_code(message: str) -> tuple:
    """
    Extracts code from:
      ```python\ncode\n```  → ("code", "python")
      ```\ncode\n```        → ("code", "")
      plain text            → (message, "")
    """
    # Fenced code block with language tag
    match = re.search(r'```(\w*)\n(.*?)```', message, re.DOTALL)
    if match:
        lang = match.group(1).strip()
        code = match.group(2).strip()
        return code, lang

    # Fenced code block without language tag
    match = re.search(r'```(.*?)```', message, re.DOTALL)
    if match:
        return match.group(1).strip(), ""

    # No code block markers — treat whole message as code
    return message.strip(), ""


# ---------------------------------------------------------------------------
# Safety check — scan for blocked patterns before execution
# ---------------------------------------------------------------------------
def safety_check(code: str) -> tuple:
    """Returns (is_safe, reason). is_safe=False means blocked."""
    for pattern, reason in BLOCKED_PATTERNS:
        if re.search(pattern, code):
            return False, reason
    return True, ""


# ---------------------------------------------------------------------------
# Python runner — writes to temp file, runs in subprocess with timeout
# ---------------------------------------------------------------------------
def run_python(code: str, timeout: int = MAX_EXEC_TIME) -> dict:
    """
    Runs Python code safely. Returns:
    {
        "success": bool,
        "stdout": str,
        "stderr": str,
        "exec_time": float,
        "timed_out": bool,
    }
    """
    # Write to temp file (avoids shell escaping issues)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        exec_time = time.time() - start
        stdout = result.stdout[:MAX_OUTPUT_SIZE]
        stderr = result.stderr[:MAX_OUTPUT_SIZE]
        return {
            "success":   result.returncode == 0,
            "stdout":    stdout,
            "stderr":    stderr,
            "exec_time": round(exec_time, 3),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "success":   False,
            "stdout":    "",
            "stderr":    f"⏱️ Execution timed out after {timeout}s",
            "exec_time": timeout,
            "timed_out": True,
        }
    except Exception as e:
        return {
            "success":   False,
            "stdout":    "",
            "stderr":    str(e),
            "exec_time": 0,
            "timed_out": False,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main entry point — detect language, safety check, run
# ---------------------------------------------------------------------------
def run_code(code: str, language_hint: str = "") -> dict:
    """
    Full pipeline: detect → safety check → run.
    Returns result dict with language, safety info, and execution output.
    """
    language = detect_language(code, hint=language_hint)

    is_safe, reason = safety_check(code)
    if not is_safe:
        return {
            "language": language,
            "blocked":  True,
            "reason":   reason,
            "success":  False,
            "stdout":   "",
            "stderr":   f"🚫 Blocked: {reason}",
            "exec_time": 0,
        }

    if language == "python":
        result = run_python(code)
    else:
        # Non-Python: don't run, just return unsupported message
        result = {
            "success":   False,
            "stdout":    "",
            "stderr":    f"Language '{language}' execution not supported yet (Python only).",
            "exec_time": 0,
            "timed_out": False,
        }

    return {"language": language, "blocked": False, **result}
