"""Health check and API info endpoints."""

import os

from fastapi import APIRouter

from app.core.config import USE_MOCK, VECTOR_DB_PATH

router = APIRouter(tags=["health"])


def _check_rag_enabled() -> bool:
    """Check if the RAG vector store directory exists and has data."""
    return os.path.isdir(VECTOR_DB_PATH)


@router.get("/health")
def health_check() -> dict:
    return {
        "status": "healthy",
        "mock": USE_MOCK,
        "rag_enabled": _check_rag_enabled(),
    }


@router.get("/api")
def api_info() -> dict:
    return {
        "endpoints": [
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/api", "method": "GET", "description": "List all API endpoints"},
            {"path": "/chat", "method": "POST", "description": "Knowledge Q&A chat"},
            {"path": "/chat/audio", "method": "POST", "description": "Audio-based chat"},
            {"path": "/lesson", "method": "POST", "description": "Structured lesson interaction"},
            {"path": "/exercises/{exercise_id}", "method": "GET", "description": "Download exercise PDF"},
            {"path": "/audio/upload", "method": "POST", "description": "Upload audio for analysis"},
            {"path": "/sessions/{session_id}", "method": "GET", "description": "Get session history"},
            {"path": "/sessions/{session_id}", "method": "DELETE", "description": "Clear session history"},
            {"path": "/", "method": "GET", "description": "Web UI"},
        ]
    }
