"""The shared per-request pipeline behind one deep module.

Every ``/chat`` and ``/lesson`` request runs the same pre-LLM pipeline: resolve
the era, retrieve RAG context, assemble the system prompt, and build citations.
This module owns that pipeline so the two route handlers stop inlining (and
silently diverging on) it. The single public entry point is
``resolve_request_context(...) -> RequestContext``; ``resolve_era(...)`` is the
shared era-resolution path both modes cross.

See ``CONTEXT.md`` for the RequestContext / era vocabulary. Citation building is
delegated to ``app.core.citations.build_citations`` (issue #1) — it is *not*
re-implemented here.

Lesson-specific concerns (session state, drills, audio analysis) stay in the
lesson handler: the lesson handler resolves its era with ``resolve_era``, runs
its state machine to fill the lesson prompt, and then calls
``resolve_request_context`` with that prompt as ``base_prompt``. The seam is the
shared pre-LLM pipeline, not the lesson state machine.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from app.core.era_router import keyword_classify_era, determine_dominant_era
from app.core.response_builder import (
    load_era_prompt,
    build_system_prompt_with_context,
)
from app.core.citations import build_citations

# The four valid manual-override eras. Anything else falls back to keyword
# detection (and then GENERAL).
_VALID_ERAS = {"TRUMPET_YOGA", "SUPERCHOPS", "TCE", "GENERAL"}

# A RAG retriever: (query, era) -> list of raw result dicts. Injected so tests
# can pass fixture results without ChromaDB. ``era=None`` means Auto retrieval.
Retriever = Callable[..., List[Dict]]


def _default_retrieve(query: str, era: Optional[str]) -> List[Dict]:
    """Default retriever — the real ChromaDB-backed service.

    Imported lazily so importing this module (and unit-testing it with a fake
    retriever) does not drag in chromadb / sentence-transformers.
    """
    from app.services.rag import retrieve_context

    return retrieve_context(query=query, era=era)


@dataclass
class EraResolution:
    """The outcome of resolving which era a request belongs to.

    * ``era`` — the value surfaced to the client. ``"AUTO"`` in Auto mode.
    * ``prompt_era`` — the *concrete* era to load a prompt for and drive
      topic/drill selection with. Equals ``era`` for explicit/saved eras and the
      dominant era in Auto mode (never ``"AUTO"``).
    * ``rag_results`` — the results already fetched while determining the
      dominant era in Auto mode, so the caller can reuse them instead of
      retrieving twice. ``None`` for explicit/saved eras.
    * ``is_auto`` — whether this was Auto mode.
    """

    era: str
    prompt_era: str
    rag_results: Optional[List[Dict]] = None
    is_auto: bool = False


@dataclass
class RequestContext:
    """Everything the shared pipeline produces before the LLM call.

    Maps 1:1 to the response fields both handlers fill: ``era``, ``citations``,
    ``forum_citations``, ``images``, ``rag_enabled``. ``system_prompt`` and
    ``prior_messages`` feed the LLM call; ``rag_results`` is kept raw for any
    downstream use (e.g. ``log_citation_tiers``).
    """

    era: str
    system_prompt: str
    rag_results: List[Dict]
    rag_enabled: bool
    prior_messages: List[Dict] = field(default_factory=list)
    citations: List[Dict] = field(default_factory=list)
    forum: List[Dict] = field(default_factory=list)
    images: List[Dict] = field(default_factory=list)


def resolve_era(
    text: str,
    requested_era: Optional[str],
    *,
    mode: str,
    saved_era: Optional[str] = None,
    retrieve: Optional[Retriever] = None,
) -> EraResolution:
    """Resolve the era for one request, honoring the documented precedence.

    The single era-resolution path shared by both modes, so ``/chat`` and
    ``/lesson`` cannot diverge in how they pick an era:

    * explicit override (when valid) → keyword fallback (then GENERAL);
    * ``mode == "lesson"`` additionally honors a saved-state era ahead of Auto;
    * otherwise Auto: retrieve unscoped, then take the dominant era.

    Auto mode is the only branch that retrieves — it must, to find the dominant
    era — and it hands those results back on the :class:`EraResolution` so the
    caller does not retrieve a second time.
    """
    retrieve = retrieve or _default_retrieve

    if requested_era:
        normalized = requested_era.upper().replace(" ", "_").replace("-", "_")
        era = normalized if normalized in _VALID_ERAS else (
            keyword_classify_era(text) or "GENERAL"
        )
        return EraResolution(era=era, prompt_era=era)

    if mode == "lesson" and saved_era:
        return EraResolution(era=saved_era, prompt_era=saved_era)

    # Auto mode: RAG first, then determine the era from the results.
    rag_results = retrieve(query=text, era=None)
    dominant = determine_dominant_era(rag_results)
    return EraResolution(
        era="AUTO",
        prompt_era=dominant,
        rag_results=rag_results,
        is_auto=True,
    )


def resolve_request_context(
    text: str,
    *,
    mode: str,
    requested_era: Optional[str] = None,
    saved_era: Optional[str] = None,
    resolution: Optional[EraResolution] = None,
    base_prompt: Optional[str] = None,
    prior_messages: Optional[List[Dict]] = None,
    retrieve: Optional[Retriever] = None,
) -> RequestContext:
    """Run the shared pre-LLM pipeline and return a single :class:`RequestContext`.

    Steps owned here: era resolution (unless a ``resolution`` is supplied),
    RAG retrieval (reusing Auto-mode results), prompt assembly via
    ``build_system_prompt_with_context``, and citation building via
    ``build_citations``.

    Args:
        text: the user's query.
        mode: ``"chat"`` or ``"lesson"`` — only affects era precedence.
        requested_era: the request's explicit era override, if any.
        saved_era: a lesson's saved-state era (honored ahead of Auto).
        resolution: a pre-computed :class:`EraResolution`. The lesson handler
            resolves the era up front (its state machine needs the era before it
            can build the prompt) and passes it back in so the era is resolved
            exactly once.
        base_prompt: a fully-assembled base prompt to inject RAG context into.
            Defaults to ``load_era_prompt(resolution.prompt_era)`` — the chat
            case. The lesson handler passes its state-filled lesson prompt.
        prior_messages: conversation history for the LLM call, passed through
            verbatim onto the context.
        retrieve: injectable RAG retriever (defaults to the ChromaDB service).

    Degrades gracefully: when retrieval returns nothing, ``rag_enabled`` is
    ``False`` and the empty-context note flows through exactly as before.
    """
    retrieve = retrieve or _default_retrieve

    if resolution is None:
        resolution = resolve_era(
            text, requested_era, mode=mode, saved_era=saved_era, retrieve=retrieve
        )

    # Reuse the results fetched during Auto resolution; otherwise retrieve for
    # the resolved era.
    if resolution.rag_results is not None:
        rag_results = resolution.rag_results
    else:
        rag_results = retrieve(query=text, era=resolution.era)
    rag_enabled = len(rag_results) > 0

    if base_prompt is None:
        base_prompt = load_era_prompt(resolution.prompt_era)
    system_prompt = build_system_prompt_with_context(base_prompt, rag_results)

    cs = build_citations(rag_results)

    return RequestContext(
        era=resolution.era,
        system_prompt=system_prompt,
        rag_results=rag_results,
        rag_enabled=rag_enabled,
        prior_messages=prior_messages or [],
        citations=cs.citations,
        forum=cs.forum,
        images=cs.images,
    )
