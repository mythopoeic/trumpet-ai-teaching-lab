"""Audio upload and analysis endpoint."""

import logging
import os
import sys
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

logger = logging.getLogger(__name__)

router = APIRouter(tags=["audio"])

# Ensure services/ is on sys.path for rag imports
_SERVICES_DIR = str(Path(__file__).resolve().parents[3] / "services")
if _SERVICES_DIR not in sys.path:
    sys.path.insert(0, _SERVICES_DIR)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".webm"}
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/audio/upload")
async def upload_audio(
    file: UploadFile = File(...),
    audio_type: str = Form(None),
    session_id: str = Form(None),
):
    """Upload an audio file for analysis.

    Accepts WAV, MP3, M4A, OGG, and WebM formats up to 10 MB.
    Returns audio analysis results as JSON.
    """
    # Validate file extension
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type '%s'. Accepted formats: %s"
            % (ext, ", ".join(sorted(ALLOWED_EXTENSIONS))),
        )

    # Read file content and validate size
    content = await file.read()
    if len(content) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 10 MB.",
        )

    # Save to temp file for analysis (Windows needs delete=False for librosa)
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        tmp.write(content)
        tmp.close()

        from rag.audio_analyzer import get_audio_analyzer

        analyzer = get_audio_analyzer()
        analysis = analyzer.analyze_audio(
            tmp.name, extract_notes=True, extract_characteristics=True
        )
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail="Audio analysis unavailable: %s" % str(e),
        )
    except Exception as e:
        logger.warning("Audio analysis failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Audio analysis failed: %s" % str(e),
        )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    return {
        "status": "success",
        "filename": filename,
        "audio_type": audio_type,
        "session_id": session_id,
        "analysis": analysis,
    }
