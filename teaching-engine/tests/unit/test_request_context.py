"""Unit tests for the shared per-request pipeline module.

These exercise ``resolve_request_context`` / ``resolve_era`` through their public
interface: inject a fake retriever returning fixture ``rag_results``, use the
real on-disk prompt loader, and assert on the returned ``RequestContext`` /
``EraResolution``. No endpoint, no ChromaDB, no Claude — mirroring the prior art
in ``tests/unit/test_citation_helpers.py`` and ``tests/unit/test_dp_tier_a.py``.

Run from ``teaching-engine/``:

    python -m pytest tests/unit/test_request_context.py -v
"""

from typing import Dict, List, Optional

from app.core.request_context import (
    EraResolution,
    RequestContext,
    resolve_era,
    resolve_request_context,
)


# --- Fixtures --------------------------------------------------------------


class FakeRetriever:
    """Records calls and returns canned results, keyed by nothing — every call
    returns the same fixture list. ``calls`` captures (query, era) tuples."""

    def __init__(self, results: List[Dict]):
        self.results = results
        self.calls: List[tuple] = []

    def __call__(self, query: str, era: Optional[str]) -> List[Dict]:
        self.calls.append((query, era))
        return self.results


def _book_result(filename="trumpet-yoga.pdf", era="TRUMPET_YOGA", page=12):
    return {
        "text": "book content about breathing",
        "source": filename,
        "era": era,
        "tier": "book",
        "metadata": {"filename": filename, "source_pdf": filename, "page_number": page},
    }


def _forum_result(topic="Embouchure tips", url="https://f.example/1", era="TCE"):
    return {
        "text": "forum chatter",
        "source": "trumpetherald.com",
        "era": era,
        "tier": "forum",
        "metadata": {"topic_title": topic, "url": url},
    }


# --- resolve_era: the precedence table -------------------------------------


def test_explicit_valid_era_is_normalized_and_used():
    res = resolve_era("anything", "trumpet-yoga", mode="chat", retrieve=FakeRetriever([]))
    assert isinstance(res, EraResolution)
    assert res.era == "TRUMPET_YOGA"
    assert res.prompt_era == "TRUMPET_YOGA"
    assert res.is_auto is False
    assert res.rag_results is None


def test_invalid_era_falls_back_to_keyword():
    res = resolve_era("how do I do superchops?", "bogus-era", mode="chat",
                      retrieve=FakeRetriever([]))
    assert res.era == "SUPERCHOPS"
    assert res.is_auto is False


def test_invalid_era_no_keyword_falls_back_to_general():
    res = resolve_era("just a plain question", "bogus-era", mode="chat",
                      retrieve=FakeRetriever([]))
    assert res.era == "GENERAL"


def test_no_era_goes_auto_and_takes_dominant():
    fake = FakeRetriever([_book_result(era="TCE"), _book_result(era="TCE"),
                          _book_result(era="TRUMPET_YOGA")])
    res = resolve_era("question", None, mode="chat", retrieve=fake)
    assert res.era == "AUTO"
    assert res.prompt_era == "TCE"  # dominant
    assert res.is_auto is True
    assert res.rag_results is not None
    assert fake.calls == [("question", None)]  # retrieved unscoped exactly once


def test_lesson_saved_state_era_precedes_auto():
    fake = FakeRetriever([_book_result(era="TCE")])
    res = resolve_era("question", None, mode="lesson", saved_era="SUPERCHOPS",
                      retrieve=fake)
    assert res.era == "SUPERCHOPS"
    assert res.prompt_era == "SUPERCHOPS"
    assert res.is_auto is False
    assert fake.calls == []  # saved era short-circuits before retrieval


def test_chat_ignores_saved_era_and_goes_auto():
    fake = FakeRetriever([_book_result(era="TCE")])
    res = resolve_era("question", None, mode="chat", saved_era="SUPERCHOPS",
                      retrieve=fake)
    assert res.era == "AUTO"  # chat has no saved-era branch


# --- resolve_request_context: the full pipeline ----------------------------


def test_context_has_citations_and_injected_prompt():
    fake = FakeRetriever([_book_result(), _forum_result()])
    ctx = resolve_request_context("breathing", mode="chat", requested_era="TRUMPET_YOGA",
                                  retrieve=fake)
    assert isinstance(ctx, RequestContext)
    assert ctx.era == "TRUMPET_YOGA"
    assert ctx.rag_enabled is True
    # System prompt carries the numbered RAG context built by response_builder.
    assert "Relevant source material" in ctx.system_prompt
    assert "[1]" in ctx.system_prompt
    assert "book content about breathing" in ctx.system_prompt
    # Citations were built by the CitationSet builder (book grouped, forum split).
    assert any(c.get("tier") == "book" for c in ctx.citations)
    assert any(f.get("tier") == "forum" for f in ctx.forum)
    # Explicit era retrieves scoped to that era.
    assert fake.calls == [("breathing", "TRUMPET_YOGA")]


def test_empty_retrieval_disables_rag():
    fake = FakeRetriever([])
    ctx = resolve_request_context("q", mode="chat", requested_era="GENERAL",
                                  retrieve=fake)
    assert ctx.rag_enabled is False
    assert ctx.citations == []
    assert ctx.forum == []
    assert ctx.images == []
    assert "No specific source material was found" in ctx.system_prompt


def test_auto_mode_reuses_results_without_second_retrieval():
    fake = FakeRetriever([_book_result(era="TCE"), _book_result(era="TCE")])
    ctx = resolve_request_context("q", mode="chat", requested_era=None, retrieve=fake)
    assert ctx.era == "AUTO"
    assert ctx.rag_enabled is True
    # Auto retrieval happened once (era=None); results reused for the pipeline.
    assert fake.calls == [("q", None)]


def test_lesson_uses_supplied_base_prompt_and_resolution():
    fake = FakeRetriever([_book_result(era="SUPERCHOPS")])
    res = resolve_era("q", None, mode="lesson", saved_era="SUPERCHOPS", retrieve=fake)
    ctx = resolve_request_context(
        "q", mode="lesson", resolution=res,
        base_prompt="LESSON BASE PROMPT", prior_messages=[{"role": "user", "content": "hi"}],
        retrieve=fake,
    )
    assert ctx.era == "SUPERCHOPS"
    assert ctx.system_prompt.startswith("LESSON BASE PROMPT")
    assert ctx.prior_messages == [{"role": "user", "content": "hi"}]
    # Saved-era resolution had no results, so the pipeline retrieved scoped once.
    assert fake.calls == [("q", "SUPERCHOPS")]
