"""Mock Anthropic client for local testing without an API key."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Call counter for response variety
_call_count = 0


class MockContentBlock:
    """Mimics anthropic.types.ContentBlock with .type and .text attributes."""

    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class MockMessage:
    """Mimics anthropic.types.Message with .content list."""

    def __init__(self, text: str):
        self.content = [MockContentBlock(text)]
        self.id = "msg_mock"
        self.model = "mock"
        self.role = "assistant"
        self.stop_reason = "end_turn"
        self.type = "message"


class _MockMessages:
    """Namespace object providing .create() to match client.messages.create()."""

    def create(
        self,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        system: str = "",
        messages: Optional[list] = None,
    ) -> MockMessage:
        global _call_count
        _call_count += 1

        messages = messages or []

        # Extract user text from last message
        user_text = ""
        conversation_history = None
        if messages:
            last = messages[-1]
            if last.get("role") == "user":
                user_text = last.get("content", "")
            if len(messages) > 1:
                conversation_history = messages[:-1]

        text = _generate_mock_response(system, user_text, conversation_history)
        return MockMessage(text)


class MockAnthropicClient:
    """Drop-in replacement for anthropic.Anthropic() in mock mode."""

    def __init__(self, api_key: str = ""):
        self.messages = _MockMessages()


def _generate_mock_response(
    system_prompt: str,
    user_text: str,
    conversation_history: Optional[list] = None,
) -> str:
    """Generate a contextual mock response based on system prompt and user input."""
    global _call_count

    sp_lower = system_prompt.lower()
    if "superchops" in sp_lower or "super chops" in sp_lower:
        era = "SUPERCHOPS"
    elif "tce" in sp_lower or "tongue" in sp_lower:
        era = "TCE"
    else:
        era = "TRUMPET_YOGA"

    user_lower = user_text.lower()
    is_followup = bool(conversation_history)

    if "embouchure" in user_lower or "lips" in user_lower:
        if era == "TCE":
            return (
                "[MOCK TCE] In the Tongue-Controlled Embouchure approach, the tongue "
                "plays a primary role in directing the air stream and controlling the "
                "aperture. Focus on tongue placement rather than manipulating the lips."
            )
        if era == "SUPERCHOPS":
            return (
                "[MOCK Superchops] Superchops emphasizes a forward, bunched lip "
                "position. The key is maintaining this forward position while keeping "
                "the corners firm, creating compression for powerful playing."
            )
        return (
            "[MOCK Trumpet Yoga] In Trumpet Yoga, the lips should be centered and "
            "relaxed. Think about the embouchure as a natural extension of your "
            "breathing — it should form organically."
        )

    if any(w in user_lower for w in ("breathing", "air", "breath")):
        return (
            f"[MOCK {era}] Proper breathing is fundamental. Take a full, relaxed "
            f"breath filling the lower lungs. Air support comes from the diaphragm."
        )

    if any(w in user_lower for w in ("range", "high", "register")):
        return (
            f"[MOCK {era}] Range development comes from proper air compression and "
            f"embouchure efficiency. Practice scales and lip slurs regularly."
        )

    if any(w in user_lower for w in ("practice", "exercise", "routine")):
        return (
            f"[MOCK {era}] Start with long tones, then flexibility exercises, "
            f"then etudes. Quality over quantity — 30-45 minutes of focused practice "
            f"beats hours of mindless repetition."
        )

    if any(w in user_lower for w in ("what is", "tell me about", "explain")):
        return (
            f"[MOCK {era}] {era.replace('_', ' ').title()} is one of Jerome Callet's "
            f"teaching methods, emphasizing proper fundamentals for trumpet playing."
        )

    if is_followup:
        return (
            f"[MOCK {era}] Building on our previous discussion — could you elaborate "
            f"on what aspect you'd like to explore further?"
        )

    topics = ["embouchure", "breathing", "range", "practice routine", "technique"]
    topic = topics[_call_count % len(topics)]
    return (
        f"[MOCK {era}] Thank you for your question about trumpet technique. "
        f"For this topic, consider working on {topic}. Would you like more "
        f"specific guidance?"
    )
