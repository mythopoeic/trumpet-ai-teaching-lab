"""Unit tests for the Budget cascade (``app.services.rag_budget``).

These exercise the tiered character allocation + downward redistribution
(book -> media -> forum) through its public interface, with **no ChromaDB and
no sentence-transformers** — the whole point of extracting the cascade into a
pure module. Mirrors the prior art in ``tests/unit/test_citation_helpers.py``.

Run from ``teaching-engine/``:

    python -m pytest tests/unit/test_rag_budget.py -v
"""

from app.services.rag_budget import (
    allocate_budget_cascade,
    _fit_within_budget,
)


# --- Helpers ---------------------------------------------------------------


def _chunk(n_chars: int, tag: str = "x") -> dict:
    """A result-shaped dict whose ``text`` is exactly *n_chars* long."""
    return {"text": tag * n_chars, "tier": tag}


# A clean 1000-char budget: book 600 / media 250 / forum 150.
BUDGET = dict(total_budget=1000, book_pct=0.6, media_pct=0.25, forum_pct=0.15)


# --- _fit_within_budget ----------------------------------------------------


def test_fit_takes_chunks_in_order_until_full():
    chunks = [_chunk(400), _chunk(400)]
    selected, used = _fit_within_budget(chunks, 600)
    # First fits (400); second would overflow (800 > 600) and is skipped.
    assert len(selected) == 1
    assert used == 400


def test_fit_skips_an_overflowing_chunk_but_takes_a_later_smaller_one():
    chunks = [_chunk(600), _chunk(50)]
    selected, used = _fit_within_budget(chunks, 100)
    # The 600 overflows and is skipped; the later 50 still fits. Selection is
    # not just a prefix.
    assert [len(c["text"]) for c in selected] == [50]
    assert used == 50


def test_fit_empty_budget_selects_nothing():
    selected, used = _fit_within_budget([_chunk(10)], 0)
    assert selected == []
    assert used == 0


# --- allocate_budget_cascade: split ----------------------------------------


def test_split_is_pct_of_total_when_each_tier_fills_its_share():
    # Each tier has one chunk exactly the size of its base budget.
    res = allocate_budget_cascade(
        [_chunk(600, "b")], [_chunk(250, "m")], [_chunk(150, "f")], **BUDGET
    )
    assert res.book.used == 600 and res.book.budget == 600
    assert res.media.used == 250 and res.media.budget == 250
    assert res.forum.used == 150 and res.forum.budget == 150


# --- allocate_budget_cascade: downward cascade -----------------------------


def test_unused_book_budget_flows_to_media():
    # Books use only 100 of 600 -> 500 cascades to media (250 -> 750).
    res = allocate_budget_cascade(
        [_chunk(100, "b")], [_chunk(700, "m")], [], **BUDGET
    )
    assert res.book.used == 100
    assert res.media.budget == 750  # 250 base + 500 inherited
    assert res.media.used == 700  # the 700-char chunk now fits


def test_unused_book_and_media_both_flow_to_forum():
    # Books use 100/600 (500 spills), media uses 0/(250+500=750) (750 spills),
    # forum budget becomes 150 + 750 = 900.
    res = allocate_budget_cascade(
        [_chunk(100, "b")], [], [_chunk(880, "f")], **BUDGET
    )
    assert res.media.used == 0
    assert res.forum.budget == 900  # 150 + 500 (book) + 250 (media base)
    assert res.forum.used == 880  # fits only because of the cascade


def test_no_upward_flow_media_surplus_does_not_help_books():
    # A huge book candidate must NOT be rescued by unused media/forum budget.
    res = allocate_budget_cascade(
        [_chunk(900, "b")], [], [], **BUDGET
    )
    assert res.book.used == 0  # 900 > 600 book budget, nothing below flows up


# --- allocate_budget_cascade: ordering & invariants ------------------------


def test_combined_chunks_are_book_then_media_then_forum():
    res = allocate_budget_cascade(
        [_chunk(10, "b")], [_chunk(10, "m")], [_chunk(10, "f")], **BUDGET
    )
    assert [c["tier"] for c in res.chunks] == ["b", "m", "f"]


def test_empty_everything_yields_empty_result():
    res = allocate_budget_cascade([], [], [], **BUDGET)
    assert res.chunks == []
    assert res.book.used == res.media.used == res.forum.used == 0


def test_total_used_never_exceeds_total_budget():
    # Oversupply every tier; cascade must still respect the global ceiling.
    res = allocate_budget_cascade(
        [_chunk(5000, "b")],
        [_chunk(5000, "m")],
        [_chunk(5000, "f")],
        **BUDGET,
    )
    total_used = res.book.used + res.media.used + res.forum.used
    assert total_used <= BUDGET["total_budget"]
