"""Trumpet AI Teaching Lab - FastAPI application (portfolio snapshot).

Public surface: a grounded knowledge/RAG endpoint and audio-analysis endpoints.
The stateful lesson/teaching product is excluded from this public snapshot
(see docs/portfolio-snapshot.md).
"""

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)

from dotenv import load_dotenv

load_dotenv()

# Add services/ directory to sys.path for rag/audio imports
_SERVICES_DIR = str(Path(__file__).resolve().parent.parent / "services")
if _SERVICES_DIR not in sys.path:
    sys.path.insert(0, _SERVICES_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import audio, chat, health, sessions

app = FastAPI(
    title="Trumpet AI Teaching Lab",
    description="Grounded RAG + audio-analysis API for trumpet pedagogy (portfolio snapshot)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(audio.router)


@app.get("/")
def root() -> dict:
    return {
        "name": "Trumpet AI Teaching Lab",
        "status": "ok",
        "note": (
            "Portfolio snapshot: grounded RAG + audio-analysis API. The "
            "lesson/teaching product, production prompts, and private corpus "
            "are excluded. See docs/portfolio-snapshot.md."
        ),
    }


# Mount uploads/audio for the audio endpoint's playback (runtime dir)
_UPLOADS_AUDIO_DIR = "uploads/audio"
os.makedirs(_UPLOADS_AUDIO_DIR, exist_ok=True)
app.mount("/uploads/audio", StaticFiles(directory=_UPLOADS_AUDIO_DIR), name="uploads_audio")
