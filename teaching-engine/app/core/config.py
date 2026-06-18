"""Centralized configuration for the Callet Teaching Bot.

Reads from environment variables with sensible defaults.
"""

import os

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL: str = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
USE_MOCK: bool = os.environ.get("USE_MOCK", "true").lower() == "true"

VECTOR_DB_PATH: str = os.environ.get("VECTOR_DB_PATH", "data/vector_db/")
AUDIO_UPLOAD_DIR: str = os.environ.get("AUDIO_UPLOAD_DIR", "uploads/audio/")
MAX_CONVERSATION_HISTORY: int = int(os.environ.get("MAX_CONVERSATION_HISTORY", "10"))
MAX_AUDIO_SIZE_MB: int = int(os.environ.get("MAX_AUDIO_SIZE_MB", "10"))

# RAG tiered retrieval budget
RAG_TOTAL_CONTEXT_BUDGET: int = int(os.environ.get("RAG_TOTAL_CONTEXT_BUDGET", "12000"))
RAG_BUDGET_BOOK_PCT: float = float(os.environ.get("RAG_BUDGET_BOOK_PCT", "0.60"))
RAG_BUDGET_MEDIA_PCT: float = float(os.environ.get("RAG_BUDGET_MEDIA_PCT", "0.25"))
RAG_BUDGET_FORUM_PCT: float = float(os.environ.get("RAG_BUDGET_FORUM_PCT", "0.15"))
RAG_BOOK_MAX_RESULTS: int = int(os.environ.get("RAG_BOOK_MAX_RESULTS", "20"))
RAG_MEDIA_MAX_RESULTS: int = int(os.environ.get("RAG_MEDIA_MAX_RESULTS", "10"))
RAG_FORUM_MAX_RESULTS: int = int(os.environ.get("RAG_FORUM_MAX_RESULTS", "5"))

# Lesson response validation — when True, re-call Claude if high-value section missing (Phase 2)
LESSON_REPAIR_PASS: bool = os.environ.get("LESSON_REPAIR_PASS", "false").lower() == "true"

# Debug telemetry for spit-buzz attack detection
DEBUG_SPITBUZZ: bool = os.environ.get("DEBUG_SPITBUZZ", "false").lower() == "true"

# Use attack-based segmenter (envelope peak finding) instead of silence-based segment count
USE_ATTACK_SEGMENTER: bool = os.environ.get("USE_ATTACK_SEGMENTER", "true").lower() == "true"

# Enable similarity scoring against Jerry Callet reference clips
ENABLE_SIMILARITY_SCORING: bool = os.environ.get("ENABLE_SIMILARITY_SCORING", "false").lower() == "true"

# Enable media embeds (YouTube video cards) in lesson UI
ENABLE_MEDIA_EMBEDS: bool = os.environ.get("ENABLE_MEDIA_EMBEDS", "true").lower() == "true"
