"""The Budget cascade — pure tiered character allocation for RAG retrieval.

This module owns one decision and nothing else: how a fixed character budget
is split across the source tiers (Books / Media / Forum) and how unused budget
*cascades downward* (book -> media -> forum). It deliberately has **no
ChromaDB / sentence-transformers dependency**, so the allocation rule that
defines this RAG system (budget-based, not top-K) is unit-testable without a
live vector store.

See ``CONTEXT.md`` ("Budget cascade", "Tier") and the architecture overview
section 5. The caller (``app.services.rag.retrieve_context``) is responsible
for the I/O: embedding the query and fetching per-tier candidate chunks. It
hands those candidate lists plus the budget config to
``allocate_budget_cascade`` and gets back the trimmed, ordered chunks together
with per-tier accounting for logging.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TierAllocation:
    """What one tier received from the cascade.

    ``budget`` is the *effective* budget after any upstream redistribution
    (e.g. media's budget includes unused book characters), matching the
    numbers the retrieval log reports.
    """

    chunks: list[dict]
    used: int
    budget: int


@dataclass
class BudgetCascadeResult:
    """The outcome of one cascade.

    ``chunks`` is the flat, ordered result the caller returns directly:
    books first, then media, then forum. The per-tier ``TierAllocation``s
    carry the accounting (count/used/budget) for logging and tests.
    """

    chunks: list[dict]
    book: TierAllocation
    media: TierAllocation
    forum: TierAllocation


def _fit_within_budget(chunks: list[dict], budget: int) -> tuple[list[dict], int]:
    """Greedily select, in order, the chunks whose text fits within *budget*.

    Chunks are assumed pre-sorted by relevance. A chunk that would overflow the
    budget is skipped (later, smaller chunks may still fit). Returns
    ``(selected_chunks, chars_used)``.
    """
    selected: list[dict] = []
    used = 0
    for chunk in chunks:
        chunk_len = len(chunk.get("text", ""))
        if used + chunk_len <= budget:
            selected.append(chunk)
            used += chunk_len
    return selected, used


def allocate_budget_cascade(
    book_results: list[dict],
    media_results: list[dict],
    forum_results: list[dict],
    *,
    total_budget: int,
    book_pct: float,
    media_pct: float,
    forum_pct: float,
) -> BudgetCascadeResult:
    """Allocate *total_budget* characters across tiers with downward cascade.

    The split is ``book_pct`` / ``media_pct`` / ``forum_pct`` of
    ``total_budget``. Unused characters flow strictly downward: whatever books
    leave unfilled is added to the media budget, and whatever media then leaves
    unfilled is added to the forum budget. (Forum is the bottom tier; its
    leftover is simply unused.)

    Each tier's candidate list is assumed pre-sorted by relevance. The returned
    ``chunks`` is ``books + media + forum`` in that order.
    """
    book_budget = int(total_budget * book_pct)
    media_budget = int(total_budget * media_pct)
    forum_budget = int(total_budget * forum_pct)

    # Books — top tier, no inflow.
    budgeted_books, book_used = _fit_within_budget(book_results, book_budget)
    book_unused = book_budget - book_used

    # Media — inherits unused book characters.
    media_budget += book_unused
    budgeted_media, media_used = _fit_within_budget(media_results, media_budget)
    media_unused = media_budget - media_used

    # Forum — bottom tier, inherits unused media characters.
    forum_budget += media_unused
    budgeted_forum, forum_used = _fit_within_budget(forum_results, forum_budget)

    return BudgetCascadeResult(
        chunks=budgeted_books + budgeted_media + budgeted_forum,
        book=TierAllocation(budgeted_books, book_used, book_budget),
        media=TierAllocation(budgeted_media, media_used, media_budget),
        forum=TierAllocation(budgeted_forum, forum_used, forum_budget),
    )
