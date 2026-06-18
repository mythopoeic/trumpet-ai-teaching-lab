"""Citation building — the CitationSet domain logic.

This module owns *all* citation assembly. The single public entry point is
``build_citations(rag_results) -> CitationSet``: a pure function (no I/O, no
network) called from the shared pre-LLM pipeline
(``app.core.request_context.resolve_request_context``); both route handlers
receive the result via ``RequestContext`` and pass it to their response
schemas. See ``CONTEXT.md`` for the Citation / CitationSet / Hidden source
vocabulary.

It lives in ``app.core`` (not ``app.api.routes``) because it is core domain
logic, not HTTP glue — keeping the dependency arrow pointing inward
(``core`` never imports from ``routes``).

The output dict keys are the frozen single-file-SPA contract:

* citations: ``tier``, ``source``, ``page_numbers``, ``page_number``,
  ``media_title``, ``speaker_name``, ``era`` (a given dict carries only the
  subset relevant to its tier)
* forum: ``topic``, ``url``, ``tier``
* images: ``path``, ``description``, ``source_pdf``, ``page_number``, ``era``
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Map source-filename substrings to friendly display names.
# PORTFOLIO SNAPSHOT: the production mapping for the private corpus is withheld;
# this illustrates the pattern only. A value of None marks a "hidden source" --
# ingested to inform answers but never shown as a citation.
_BOOK_DISPLAY_NAMES: Dict[str, Optional[str]] = {
    # "example-primary-source": "Example Primary Source",
    # "example-supplement": None,  # hidden source
}

# Sources that should inform answers but never appear as cited sources.
HIDDEN_SOURCES: set = set()

# Extracted-PDF image path prefix, rewritten to the /images/ static mount.
_IMAGE_PREFIX = "images/extracted/"

# Default era when a RAG result omits one (the safe behaviour; previously
# /chat read the raw subscript and /lesson defaulted — see issue #1).
_DEFAULT_ERA = "GENERAL"


@dataclass
class CitationSet:
    """The result of building citations for one answer.

    ``citations`` is book (grouped) + media merged into one list, each dict
    discriminated by its ``tier`` key. ``forum`` and ``images`` are kept
    separate. These map 1:1 to the SPA response fields ``citations``,
    ``forum_citations``, and ``images``.
    """

    citations: List[Dict] = field(default_factory=list)
    forum: List[Dict] = field(default_factory=list)
    images: List[Dict] = field(default_factory=list)


def get_book_display_name(metadata: Dict) -> Optional[str]:
    """Return a human-readable book title from RAG result metadata.

    Checks the 'filename' and 'source_pdf' metadata fields against
    ``_BOOK_DISPLAY_NAMES``.  Returns ``None`` for books that should
    never appear as cited sources (e.g. Balanced Embouchure).
    Falls back to a cleaned-up filename or ``'pdf'``.
    """
    filename_lower = (
        metadata.get("filename") or metadata.get("source_pdf") or ""
    ).lower()

    if not filename_lower:
        return "pdf"

    for key, display_name in _BOOK_DISPLAY_NAMES.items():
        if key in filename_lower:
            return display_name  # None means "hide this source"

    # No match — clean up the raw filename
    cleaned = filename_lower.replace(".pdf", "").replace("-", " ").strip()
    return cleaned.title() if cleaned else "pdf"


def build_citations(rag_results: List[Dict]) -> CitationSet:
    """Build a :class:`CitationSet` from a list of RAG-result dicts.

    Pure function: no I/O, no network, and never raises on a malformed result
    dict — every field access is a defensive ``.get`` so a missing piece of
    metadata degrades to a default rather than a 500.

    Owns the whole pipeline: tier-splitting, book grouping with page sorting,
    media dedup, forum dedup, hidden-source filtering, and image-card building.
    """
    book_results: List[Dict] = []
    media_results: List[Dict] = []
    forum_results: List[Dict] = []
    image_results: List[Dict] = []
    for result in rag_results:
        tier = result.get("tier", "forum")
        if tier in ("book", "supplement"):
            book_results.append(result)
        elif tier == "media":
            media_results.append(result)
        elif tier == "image":
            image_results.append(result)
        else:
            forum_results.append(result)

    citations = _build_book_citations(book_results) + _build_media_citations(media_results)

    return CitationSet(
        citations=citations,
        forum=_build_forum_citations(forum_results),
        images=_build_image_cards(image_results),
    )


def _build_book_citations(book_results: List[Dict]) -> List[Dict]:
    """Group book/supplement citations by display name with sorted page lists.

    Hidden sources are dropped. Pages are deduplicated and sorted Roman
    numerals first (insertion order), then Arabic numbers ascending.
    """
    # Group by display name in one pass, skipping hidden sources and
    # accumulating each chunk's page numbers. A source's era is taken from the
    # first chunk that contributes it.
    grouped: Dict[str, Dict] = {}
    for result in book_results:
        meta = result.get("metadata") or {}
        display_name = get_book_display_name(meta)
        if display_name is None:
            continue  # supplementary book — never shown as a source
        if display_name not in grouped:
            grouped[display_name] = {
                "source": display_name,
                "era": result.get("era", _DEFAULT_ERA),
                "tier": "book",
                "page_numbers": [],
            }
        pn = meta.get("page_number") or meta.get("page")
        if pn:
            for p in str(pn).split(","):
                p = p.strip()
                if p:
                    grouped[display_name]["page_numbers"].append(p)

    # Deduplicate while preserving insertion order, then sort:
    # Roman numerals first, then Arabic numbers sorted numerically.
    for g in grouped.values():
        pages = list(dict.fromkeys(g["page_numbers"]))
        nums = sorted([p for p in pages if p.isdigit()], key=int)
        romans = [p for p in pages if not p.isdigit()]
        g["page_numbers"] = romans + nums

    return list(grouped.values())


def _build_media_citations(media_results: List[Dict]) -> List[Dict]:
    """Build media citations, deduplicated by title."""
    citations: List[Dict] = []
    seen_media: set = set()
    for result in media_results:
        meta = result.get("metadata") or {}
        title = meta.get("media_title", "")
        if title in seen_media:
            continue
        seen_media.add(title)
        citations.append({
            "source": "media",
            "era": result.get("era", _DEFAULT_ERA),
            "tier": "media",
            "media_title": title,
            "speaker_name": meta.get("speaker_name", ""),
        })
    return citations


def _build_forum_citations(forum_results: List[Dict]) -> List[Dict]:
    """Build forum citations, deduplicated by url.

    Citations without a url are always included. Carries only the keys the SPA
    reads: ``topic`` (when present), ``url`` (when present), and ``tier``.
    """
    citations: List[Dict] = []
    seen_urls: set = set()

    for result in forum_results:
        meta = result.get("metadata") or {}
        citation: Dict = {"tier": "forum"}
        topic = meta.get("topic_title")
        if topic:
            citation["topic"] = topic
        url = meta.get("url")
        if url:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            citation["url"] = url
        citations.append(citation)

    return citations


def _build_image_cards(image_results: List[Dict]) -> List[Dict]:
    """Build at most 3 image cards, rewriting extracted-PDF paths to /images/."""
    images: List[Dict] = []
    for result in image_results[:3]:
        meta = result.get("metadata") or {}
        raw_path = (meta.get("image_path", "") or "").replace("\\", "/")
        if raw_path.startswith(_IMAGE_PREFIX):
            url_path = "/images/" + raw_path[len(_IMAGE_PREFIX):]
        else:
            url_path = "/images/" + raw_path
        images.append({
            "path": url_path,
            "description": result.get("text", ""),
            "era": result.get("era", _DEFAULT_ERA),
            "page_number": meta.get("page_number", 0),
            "source_pdf": meta.get("source_pdf", ""),
        })
    return images
