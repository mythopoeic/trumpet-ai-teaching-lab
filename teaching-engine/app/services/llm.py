"""LLM service with mock and Anthropic API support."""

import logging
import time
from typing import Generator, Optional

from app.core.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    USE_MOCK,
)

logger = logging.getLogger(__name__)


def generate_response(
    system_prompt: str,
    user_text: str,
    conversation_history: Optional[list] = None,
) -> str:
    """Generate a response using mock or Anthropic API.

    Args:
        system_prompt: The era-specific system prompt.
        user_text: The user's current message.
        conversation_history: Optional list of prior messages, each with
            ``{"role": "user"|"assistant", "content": "..."}``.

    Returns:
        The assistant's text response.

    Raises:
        RuntimeError: If the API call fails or key is missing.
    """
    if USE_MOCK:
        return _mock_response(system_prompt, user_text, conversation_history)
    return _anthropic_response(system_prompt, user_text, conversation_history)


def generate_response_stream(
    system_prompt: str,
    user_text: str,
    conversation_history: Optional[list] = None,
) -> Generator[str, None, None]:
    """Generate a streaming response using mock or Anthropic API.

    Yields text chunks as strings. For real API, streams via
    Anthropic's messages.stream(). For mock, yields words with delays.

    Args:
        system_prompt: The era-specific system prompt.
        user_text: The user's current message.
        conversation_history: Optional list of prior messages.

    Yields:
        Text chunks as strings.
    """
    if USE_MOCK:
        yield from _mock_response_stream(system_prompt, user_text, conversation_history)
    else:
        yield from _anthropic_response_stream(system_prompt, user_text, conversation_history)


def _mock_response_stream(
    system_prompt: str,
    user_text: str,
    conversation_history: Optional[list] = None,
) -> Generator[str, None, None]:
    """Stream mock response by yielding individual words with delays."""
    full_text = _mock_response(system_prompt, user_text, conversation_history)
    words = full_text.split(" ")
    for i, word in enumerate(words):
        if i > 0:
            yield " " + word
        else:
            yield word
        time.sleep(0.05)


def _anthropic_response_stream(
    system_prompt: str,
    user_text: str,
    conversation_history: Optional[list] = None,
) -> Generator[str, None, None]:
    """Stream response from Anthropic API using messages.stream()."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Set it in your environment or .env file."
        )

    from anthropic import Anthropic

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    messages = []
    if conversation_history:
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_text})

    try:
        with client.messages.stream(
            model=ANTHROPIC_MODEL,
            system=system_prompt,
            max_tokens=2048,
            temperature=0.3,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as exc:
        raise RuntimeError(
            f"Anthropic API streaming call failed: {exc}. Check your API key and model ID."
        ) from exc


def _mock_response(
    system_prompt: str,
    user_text: str,
    conversation_history: Optional[list] = None,
) -> str:
    from app.services.mock_anthropic import MockAnthropicClient

    client = MockAnthropicClient()

    messages = []
    if conversation_history:
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_text})

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=2048,
        temperature=0.3,
        system=system_prompt,
        messages=messages,
    )

    return response.content[0].text


def _anthropic_response(
    system_prompt: str,
    user_text: str,
    conversation_history: Optional[list] = None,
) -> str:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Set it in your environment or .env file."
        )

    from anthropic import Anthropic

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    messages = []
    if conversation_history:
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_text})

    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            system=system_prompt,
            max_tokens=2048,
            temperature=0.3,
            messages=messages,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Anthropic API call failed: {exc}. Check your API key and model ID."
        ) from exc

    blocks = response.content
    text = "".join(b.text for b in blocks if b.type == "text")
    return text.strip()
