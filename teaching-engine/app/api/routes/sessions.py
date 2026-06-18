"""Session management endpoints."""

from fastapi import APIRouter

from app.services.history import conversation_history

router = APIRouter(tags=["sessions"])


@router.get("/sessions")
def list_sessions() -> dict:
    sessions = conversation_history.list_sessions()
    return {"sessions": sessions, "count": len(sessions)}


@router.post("/sessions/{session_id}/clear")
def clear_session(session_id: str) -> dict:
    conversation_history.clear(session_id)
    return {"status": "cleared", "session_id": session_id}
