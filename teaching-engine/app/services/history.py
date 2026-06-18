"""In-memory conversation history service."""

from datetime import datetime, timezone
from typing import Optional


class ConversationHistory:
    """Stores conversation messages in-memory, keyed by session_id."""

    def __init__(self) -> None:
        self._store: dict[str, list[dict]] = {}

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> None:
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        })

    def get_messages(self, session_id: str, limit: int = 10) -> list[dict]:
        """Return the last *limit* messages for a session in chronological order."""
        messages = self._store.get(session_id, [])
        return messages[-limit:]

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        return list(self._store.keys())


# Singleton instance used across the application
conversation_history = ConversationHistory()
