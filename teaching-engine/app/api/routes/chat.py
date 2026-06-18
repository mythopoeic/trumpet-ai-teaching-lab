"""Chat endpoints."""

import json
import logging
import os
import uuid
import wave
from typing import Tuple

logger = logging.getLogger(__name__)

from fastapi import APIRouter, HTTPException, Query, Request, UploadFile, File
from starlette.responses import StreamingResponse

from app.core.config import MAX_CONVERSATION_HISTORY, AUDIO_UPLOAD_DIR, MAX_AUDIO_SIZE_MB
from app.core.response_builder import (
    load_era_prompt,
    format_audio_analysis_summary,
    build_audio_feedback_prompt,
)
from app.core.request_context import resolve_request_context, RequestContext
from app.models.schemas import ChatRequest, ChatResponse, AudioUploadResponse
from app.services.history import conversation_history
from app.services.llm import generate_response, generate_response_stream
from app.services.audio import analyze_audio
from app.services.rag import log_citation_tiers

router = APIRouter(tags=["chat"])

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".webm"}
ALLOWED_MIME_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mpeg", "audio/mp3",
    "audio/webm",
}


def _get_audio_metadata(file_path: str, ext: str) -> dict:
    """Extract duration and sample rate from an audio file."""
    if ext == ".wav":
        with wave.open(file_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            return {"duration_seconds": round(duration, 2), "sample_rate": rate}

    # For MP3/WebM, try librosa if available
    try:
        import librosa  # type: ignore
        y, sr = librosa.load(file_path, sr=None)
        duration = len(y) / float(sr)
        return {"duration_seconds": round(duration, 2), "sample_rate": int(sr)}
    except ImportError:
        # librosa not installed — return defaults; analysis will happen in US-013
        return {"duration_seconds": 0.0, "sample_rate": 0}


def _prepare_chat_context(req: ChatRequest) -> Tuple[str, RequestContext]:
    """Pre-LLM work shared by streaming and non-streaming paths.

    Era routing, RAG retrieval, prompt assembly, and citation building all live
    in ``app.core.request_context``; this handler only owns the chat-specific
    conversation-history side effects. Returns ``(session_id, RequestContext)``.
    """
    session_id = req.session_id or str(uuid.uuid4())

    # Store the user message, then build the prior-turns history for the LLM
    # (excluding the message we just added).
    conversation_history.add_message(session_id, "user", req.text)
    history_messages = conversation_history.get_messages(
        session_id, limit=MAX_CONVERSATION_HISTORY
    )
    prior_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in history_messages[:-1]
    ]

    ctx = resolve_request_context(
        req.text,
        mode="chat",
        requested_era=req.era,
        prior_messages=prior_messages,
    )
    return session_id, ctx


@router.post("/chat")
def chat(
    req: ChatRequest,
    request: Request,
    stream: bool = Query(False),
):
    # Validate text is not empty
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Missing or empty text field")

    # Check if streaming is requested via query param or Accept header
    accept = request.headers.get("accept", "")
    wants_stream = stream or "text/event-stream" in accept

    session_id, ctx = _prepare_chat_context(req)

    if wants_stream:
        return _stream_chat_response(req, session_id, ctx)

    # Non-streaming path (backward compatible)
    answer = generate_response(
        system_prompt=ctx.system_prompt,
        user_text=req.text,
        conversation_history=ctx.prior_messages or None,
    )

    log_citation_tiers(answer, ctx.rag_results)
    conversation_history.add_message(session_id, "assistant", answer)

    return ChatResponse(
        answer=answer,
        era=ctx.era,
        citations=ctx.citations,
        forum_citations=ctx.forum,
        images=ctx.images,
        mode="TEXT_ONLY",
        rag_enabled=ctx.rag_enabled,
    )


def _stream_chat_response(req: ChatRequest, session_id: str, ctx: RequestContext):
    """Return a StreamingResponse that yields SSE events."""

    def event_generator():
        full_text = []

        for chunk in generate_response_stream(
            system_prompt=ctx.system_prompt,
            user_text=req.text,
            conversation_history=ctx.prior_messages or None,
        ):
            full_text.append(chunk)
            yield "data: " + json.dumps({"token": chunk}) + "\n\n"

        # Store the complete response in conversation history
        answer = "".join(full_text)
        log_citation_tiers(answer, ctx.rag_results)
        conversation_history.add_message(session_id, "assistant", answer)

        # Send final done event with metadata
        done_event = {
            "done": True,
            "era": ctx.era,
            "citations": ctx.citations,
            "forum_citations": ctx.forum,
            "images": ctx.images,
        }
        yield "data: " + json.dumps(done_event) + "\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/chat/audio", response_model=AudioUploadResponse)
async def chat_audio(file: UploadFile = File(...)) -> AudioUploadResponse:
    # Validate file extension
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Accepted formats: WAV, MP3, WebM",
        )

    # Validate MIME type
    content_type = file.content_type or ""
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid MIME type '{content_type}'. Accepted formats: WAV, MP3, WebM",
        )

    # Read file content and check size
    content = await file.read()
    max_bytes = MAX_AUDIO_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_AUDIO_SIZE_MB}MB.",
        )

    # Ensure upload directory exists
    os.makedirs(AUDIO_UPLOAD_DIR, exist_ok=True)

    # Save with unique filename
    file_id = str(uuid.uuid4())
    save_filename = f"{file_id}{ext}"
    save_path = os.path.join(AUDIO_UPLOAD_DIR, save_filename)

    with open(save_path, "wb") as f:
        f.write(content)

    # Extract metadata
    metadata = _get_audio_metadata(save_path, ext)

    # Run audio analysis
    analysis = None
    error_msg = None
    try:
        analysis = analyze_audio(save_path)
        # Use analysis duration/sample_rate if metadata extraction returned zeros
        if metadata["duration_seconds"] == 0.0 and analysis.get("duration"):
            metadata["duration_seconds"] = round(analysis["duration"], 2)
    except ValueError as exc:
        error_msg = str(exc)
    except (ImportError, Exception) as exc:
        logger.warning("Audio analysis failed: %s", exc)
        error_msg = f"Audio analysis failed: {exc}"

    # Generate LLM interpretation of analysis results
    answer = None
    era = "TRUMPET_YOGA"  # Default era for audio feedback
    if analysis is not None:
        try:
            analysis_summary = format_audio_analysis_summary(analysis)
            base_prompt = load_era_prompt(era)
            system_prompt = build_audio_feedback_prompt(base_prompt, analysis_summary)
            answer = generate_response(
                system_prompt=system_prompt,
                user_text="Please analyze my trumpet recording and give me feedback.",
            )
        except Exception as exc:
            logger.warning("LLM interpretation failed: %s", exc)
            if error_msg:
                error_msg += f"; LLM interpretation also failed: {exc}"
            else:
                error_msg = f"Audio analyzed but LLM interpretation failed: {exc}"
    elif error_msg:
        # Analysis failed — provide a helpful error message as the answer
        answer = (
            f"I wasn't able to fully analyze your recording: {error_msg}. "
            "Please try uploading a clearer recording of at least 0.5 seconds "
            "with audible trumpet playing."
        )

    return AudioUploadResponse(
        file_id=file_id,
        duration_seconds=metadata["duration_seconds"],
        format=ext.lstrip(".").upper(),
        sample_rate=metadata["sample_rate"],
        analysis=analysis,
        answer=answer,
        era=era,
        mode="AUDIO",
        error=error_msg,
        audio_url="/uploads/audio/%s" % save_filename,
    )
