"""Load era-specific system prompts and build response context."""

import os
from typing import Any, Dict, Optional

# Prompt directory relative to the teaching-engine root
_PROMPT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "teachers", "jerome-callet", "prompts"
)

_ERA_PROMPT_FILES = {
    "TRUMPET_YOGA": "trumpet-yoga.md",
    "SUPERCHOPS": "superchops.md",
    "TCE": "tce.md",
    "GENERAL": "general.md",
}

_SHARED_KNOWLEDGE_FILE = "shared-knowledge.md"
_RELATED_METHODS_FILE = "related-methods.md"
_LESSON_PROMPT_FILE = "lesson.md"


def _read_prompt_file(filename: str) -> str:
    """Read a prompt file from the prompt dir; empty string if missing."""
    path = os.path.normpath(os.path.join(_PROMPT_DIR, filename))
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def _load_shared_knowledge() -> str:
    """Load the shared-knowledge.md content that prepends all era prompts.

    This file carries the seamless-presentation rule (never says "era", never
    names a method/phase unprompted, presents guidance as Callet's approach,
    with internal era-separation preserved) so every mode and era inherits it.
    """
    return _read_prompt_file(_SHARED_KNOWLEDGE_FILE)


def _load_related_methods() -> str:
    """Load the curated related-methods reference (adjacent, non-Callet teachers).

    Used only when the student asks about one of these names; it never changes
    the active era. Loaded into the assembled prompt in both Knowledge and
    Lesson modes so the bot can answer accurately and point students outward.
    """
    return _read_prompt_file(_RELATED_METHODS_FILE)


def load_shared_preamble() -> str:
    """Shared prompt knowledge inherited by every mode and era.

    Combines the seamless-presentation rule / shared knowledge with the
    related-methods reference. Prepended to both the era prompts (Knowledge
    mode) and the lesson prompt (Lesson mode).
    """
    parts = [p for p in (_load_shared_knowledge(), _load_related_methods()) if p]
    return "\n\n".join(parts)


def _prepend_preamble(body: str) -> str:
    """Prepend the shared preamble to a prompt body, skipping it when empty."""
    preamble = load_shared_preamble()
    if preamble:
        return preamble + "\n\n" + body
    return body


def load_era_prompt(era: str) -> str:
    """Return the system prompt for the given era, prepended with shared knowledge.

    Falls back to TRUMPET_YOGA prompt for unknown eras.
    """
    filename = _ERA_PROMPT_FILES.get(era)
    if filename is None:
        # Unknown era — fall back to trumpet yoga
        filename = _ERA_PROMPT_FILES["TRUMPET_YOGA"]

    path = os.path.join(_PROMPT_DIR, filename)
    path = os.path.normpath(path)

    with open(path, "r", encoding="utf-8") as f:
        era_prompt = f.read().strip()

    return _prepend_preamble(era_prompt)


def load_lesson_prompt() -> str:
    """Load the lesson.md system prompt, prepended with shared knowledge.

    The shared preamble (seamless-presentation rule + related-methods
    reference) is prepended so Lesson mode inherits the same seamless
    presentation and related-method knowledge as Knowledge mode.
    """
    return _prepend_preamble(_read_prompt_file(_LESSON_PROMPT_FILE))


def _format_book_label(index: int, result: dict) -> str:
    """Build a citation label for a book/supplement result with page number."""
    source = result.get("source", "unknown")
    era = result.get("era", "GENERAL")
    meta = result.get("metadata", {})
    page = meta.get("page_number") or meta.get("page")
    label = f"[{index}] ({source}"
    if page:
        label += f", p. {page}"
    label += f", {era})"
    return label


def _format_forum_label(index: int, result: dict) -> str:
    """Build a citation label for a forum result with topic title."""
    source = result.get("source", "unknown")
    era = result.get("era", "GENERAL")
    meta = result.get("metadata", {})
    topic_title = meta.get("topic_title", "")
    label = f"[{index}] ({source}"
    if topic_title:
        label += f", \"{topic_title}\""
    label += f", {era})"
    return label


def _format_media_label(index: int, result: dict) -> str:
    """Build a citation label for a media result with title and speaker."""
    era = result.get("era", "GENERAL")
    meta = result.get("metadata", {})
    media_title = meta.get("media_title", "unknown media")
    speaker_name = meta.get("speaker_name", "")
    label = f"[{index}] ({media_title}"
    if speaker_name and speaker_name != "UNKNOWN":
        label += f", {speaker_name} speaking"
    label += f", {era})"
    return label


def build_system_prompt_with_context(
    base_prompt: str,
    rag_results: Optional[list[dict]] = None,
) -> str:
    """Combine the era prompt with RAG-retrieved context.

    Results are organized by tier: books are primary source material,
    recordings are supporting material, and forum posts are community
    discussion.  The LLM instruction tells the model to base answers on
    books first.

    If *rag_results* is empty or None, appends a note telling the LLM
    that no specific source material was found.
    """
    if not rag_results:
        return (
            base_prompt
            + "\n\nNo specific source material was found. "
            "Answer based on your knowledge but note this to the user."
        )

    # Separate results by tier
    book_results: list[dict] = []
    media_results: list[dict] = []
    forum_results: list[dict] = []
    for result in rag_results:
        tier = result.get("tier", "forum")
        if tier in ("book", "supplement"):
            book_results.append(result)
        elif tier == "media":
            media_results.append(result)
        elif tier == "forum":
            forum_results.append(result)
        # image results are not included in the text context

    # Build numbered context sections
    index = 1
    sections: list[str] = []

    if book_results:
        book_parts: list[str] = []
        for result in book_results:
            label = _format_book_label(index, result)
            book_parts.append(f"{label}\n{result.get('text', '')}")
            index += 1
        sections.append(
            "### Primary Source Material (Books)\n\n"
            + "\n\n".join(book_parts)
        )

    if media_results:
        media_parts: list[str] = []
        for result in media_results:
            label = _format_media_label(index, result)
            media_parts.append(f"{label}\n{result.get('text', '')}")
            index += 1
        sections.append(
            "### Supporting Material (Recordings)\n\n"
            + "\n\n".join(media_parts)
        )

    if forum_results:
        forum_parts: list[str] = []
        for result in forum_results:
            label = _format_forum_label(index, result)
            forum_parts.append(f"{label}\n{result.get('text', '')}")
            index += 1
        sections.append(
            "### Community Discussion (Forum)\n\n"
            + "\n\n".join(forum_parts)
        )

    # Build instruction based on available sources
    has_books_or_media = book_results or media_results
    if has_books_or_media:
        instruction = (
            "IMPORTANT: Base your answer on the book content below. Only reference "
            "recordings or forum posts to supplement gaps in the book material. "
            "If book content covers the topic, do not cite other sources."
        )
    else:
        instruction = (
            "No book or media sources were found for this topic. The following "
            "content comes from community discussion. Note to the student that "
            "this is based on forum discussion rather than Callet's published "
            "materials."
        )

    context_block = "\n\n".join(sections)
    return (
        base_prompt
        + "\n\n" + instruction
        + "\n\nRelevant source material:\n\n"
        + context_block
    )


def format_audio_analysis_summary(analysis: Dict[str, Any]) -> str:
    """Format audio analysis results into a readable summary for the LLM."""
    parts: list[str] = []

    # Duration and tempo
    duration = analysis.get("duration")
    tempo = analysis.get("tempo")
    if duration:
        parts.append(f"Recording duration: {duration:.1f} seconds")
    if tempo:
        parts.append(f"Detected tempo: {tempo:.0f} BPM")

    # Pitch data
    pitch = analysis.get("pitch_data", {})
    note_count = pitch.get("note_count", 0)
    notes = pitch.get("notes", [])
    if note_count:
        parts.append(f"Notes detected: {note_count}")
    if notes:
        note_names = [n.get("note_name", "") for n in notes[:10] if n.get("note_name")]
        if note_names:
            parts.append(f"Notes played: {', '.join(note_names)}")
        avg_confidence = sum(n.get("confidence", 0) for n in notes) / len(notes)
        parts.append(f"Average pitch confidence: {avg_confidence:.1%}")

    # Tone quality
    tone = analysis.get("tone_quality", {})
    centroid = tone.get("spectral_centroid", {})
    if isinstance(centroid, dict) and "mean" in centroid:
        parts.append(f"Spectral centroid (brightness): {centroid['mean']:.0f} Hz")
    hnr = tone.get("harmonic_to_noise_ratio", {})
    if isinstance(hnr, dict) and "mean" in hnr:
        parts.append(f"Harmonic-to-noise ratio: {hnr['mean']:.1f} dB")

    # Compression metrics
    compression = analysis.get("compression_metrics", {})
    rms = compression.get("rms_energy", {})
    if isinstance(rms, dict) and "mean" in rms:
        parts.append(f"RMS energy (loudness): {rms['mean']:.4f}")
    attack = compression.get("attack_time", {})
    if isinstance(attack, dict) and "mean" in attack:
        parts.append(f"Attack time: {attack['mean']:.3f} s")

    # Buzz quality
    buzz = analysis.get("buzz_quality", {})
    sawtooth = buzz.get("sawtooth_similarity", {})
    if isinstance(sawtooth, dict) and "mean" in sawtooth:
        parts.append(f"Sawtooth similarity (buzz): {sawtooth['mean']:.3f}")
    flatness = buzz.get("spectral_flatness", {})
    if isinstance(flatness, dict) and "mean" in flatness:
        parts.append(f"Spectral flatness: {flatness['mean']:.4f}")

    if not parts:
        return "Audio analysis completed but produced no measurable results."

    return "Audio Analysis Results:\n" + "\n".join(f"- {p}" for p in parts)


def build_audio_feedback_prompt(base_prompt: str, analysis_summary: str) -> str:
    """Build a system prompt for LLM interpretation of audio analysis.

    Combines the era-specific prompt with the analysis summary and
    instructions to provide actionable teaching feedback.
    """
    return (
        base_prompt
        + "\n\nThe student has submitted a trumpet recording for analysis. "
        "Review the following analysis data and provide conversational, "
        "actionable teaching feedback. Focus on what the student is doing well "
        "and what they can improve. Be encouraging but honest.\n\n"
        + analysis_summary
    )
