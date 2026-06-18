"""Pydantic models for API request/response schemas.

This module also re-exports ``DPDetectionResult`` from
``services.audio.dp_schema`` so the API layer can reference the unified
DP-detection schema without each route file reaching across the
service-to-app import boundary directly. The TypedDict itself lives
under ``services/`` because the schema is owned by the detector code,
not by the API surface.
"""

from typing import Optional

from pydantic import BaseModel

# services/ is added to sys.path at app startup (see app/main.py); this
# import follows the project's bare-package convention.
from audio.dp_schema import DPDetectionResult


class ChatRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    era: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    era: str
    citations: list[dict] = []
    forum_citations: list[dict] = []
    images: list[dict] = []
    mode: str = "TEXT_ONLY"
    rag_enabled: bool = False


class LessonRequest(BaseModel):
    text: str
    session_id: str
    era: Optional[str] = None
    audio_data: Optional[str] = None


class LessonResponse(BaseModel):
    answer: str
    era: str
    citations: list = []
    forum_citations: list = []
    mode: str = "LESSON"
    rag_enabled: bool = False
    lesson_state: dict = {}
    exercises: Optional[list] = None
    lesson_thread: Optional[dict] = None
    audio_url: Optional[str] = None
    dp_detection: Optional[dict] = None


class ExerciseInfo(BaseModel):
    id: str
    book: str
    display_name: str
    pdf_url: Optional[str] = None
    image_url: Optional[str] = None
    section_pdf_url: Optional[str] = None
    description: Optional[str] = None
    citation: Optional[str] = None
    page_ref: Optional[str] = None
    attribution: Optional[dict] = None


class AudioUploadResponse(BaseModel):
    file_id: str
    duration_seconds: float
    format: str
    sample_rate: int
    analysis: Optional[dict] = None
    answer: Optional[str] = None
    era: Optional[str] = None
    mode: str = "AUDIO"
    error: Optional[str] = None
    audio_url: Optional[str] = None


__all__ = [
    'ChatRequest',
    'ChatResponse',
    'LessonRequest',
    'LessonResponse',
    'ExerciseInfo',
    'AudioUploadResponse',
    'DPDetectionResult',
]
