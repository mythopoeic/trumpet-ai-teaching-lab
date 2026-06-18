"""RAG retrieval service — queries ChromaDB for relevant Callet teaching content."""

from __future__ import annotations

import logging
import re
from typing import Optional, Union

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import (
    RAG_BOOK_MAX_RESULTS,
    RAG_BUDGET_BOOK_PCT,
    RAG_BUDGET_FORUM_PCT,
    RAG_BUDGET_MEDIA_PCT,
    RAG_FORUM_MAX_RESULTS,
    RAG_MEDIA_MAX_RESULTS,
    RAG_TOTAL_CONTEXT_BUDGET,
    VECTOR_DB_PATH,
)
from app.services.rag_budget import allocate_budget_cascade

logger = logging.getLogger(__name__)

# Lazy-initialised globals (heavy objects created once on first call)
_client: Optional[chromadb.ClientAPI] = None
_collection: Optional[chromadb.Collection] = None
_embedder: Optional[SentenceTransformer] = None

COLLECTION_NAME = "callet_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _get_collection() -> chromadb.Collection:
    """Return (and lazily create) the ChromaDB collection handle."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        _collection = _client.get_collection(name=COLLECTION_NAME)
        logger.info(
            "Connected to ChromaDB collection '%s' (%d documents)",
            COLLECTION_NAME,
            _collection.count(),
        )
    return _collection


def _get_embedder() -> SentenceTransformer:
    """Return (and lazily create) the sentence-transformer model."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Loaded embedding model '%s'", EMBEDDING_MODEL)
    return _embedder


def _source_type_to_type(source_type: str) -> str:
    """Map source_type metadata to a simple type label."""
    if source_type == "image":
        return "image"
    elif source_type == "pdf":
        return "pdf"
    elif source_type == "media":
        return "media"
    else:
        return "forum"


# Book name patterns for tier classification (case-insensitive partial match)
_CALLET_BOOKS = [
    "trumpet-yoga", "superchops", "trumpet-secrets",
    "brass-power", "beyond-arban", "master-superchops",
]
_SUPPLEMENT_BOOKS = ["balanced-embouchure", "smiley"]


def _classify_tier(metadata: dict) -> str:
    """Classify a RAG result into an authority tier based on its metadata.

    Returns one of: 'book', 'media', 'forum', 'image'.
    Supplements (balanced-embouchure, smiley) are merged into 'book'.
    """
    source_type = metadata.get("source_type", None)
    source = metadata.get("source", "")

    if source_type == "image":
        return "image"

    if source_type == "media" or source == "media":
        return "media"

    if source_type == "pdf" or source == "pdf":
        # Check source_pdf and filename for book name patterns
        source_pdf = (metadata.get("source_pdf", "") or "").lower()
        filename = (metadata.get("filename", "") or "").lower()
        name_to_check = source_pdf or filename
        for pattern in _CALLET_BOOKS:
            if pattern in name_to_check:
                return "book"
        for pattern in _SUPPLEMENT_BOOKS:
            if pattern in name_to_check:
                return "book"
        # Unknown PDF defaults to book tier
        return "book"

    # Everything else (forum, None, unknown) is forum tier
    return "forum"


_TIER_PRIORITY = {"book": 0, "media": 1, "forum": 2, "image": 3}


def _tier_sort_key(result: dict) -> tuple:
    """Sort key for tier-based ordering: tier priority first, then distance."""
    tier = result.get("tier", "forum")
    distance = result.get("metadata", {}).get("distance", 999.0)
    return (_TIER_PRIORITY.get(tier, 3), distance)


def _parse_results(documents: list, metadatas: list, distances: list) -> list[dict]:
    """Convert raw ChromaDB results into structured dicts."""
    output: list[dict] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        source_type = meta.get("source_type", None) or meta.get("source", "unknown")
        output.append(
            {
                "text": doc,
                "source": meta.get("source", meta.get("source_type", "unknown")),
                "era": meta.get("era", "GENERAL"),
                "type": _source_type_to_type(source_type),
                "tier": _classify_tier(meta),
                "metadata": {
                    "source_type": source_type,
                    "topic_title": meta.get("topic_title", ""),
                    "url": meta.get("url", ""),
                    "page": meta.get("page", ""),
                    "section": meta.get("section", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "distance": dist,
                    "image_path": meta.get("image_path", ""),
                    "source_pdf": meta.get("source_pdf", ""),
                    "filename": meta.get("filename", ""),
                    "page_number": meta.get("page_number", 0),
                    "media_title": meta.get("media_title", ""),
                    "media_type": meta.get("media_type", ""),
                    "media_url": meta.get("media_url", ""),
                    "speaker_name": meta.get("speaker_name", ""),
                    "is_callet_speaking": meta.get("is_callet_speaking", False),
                },
            }
        )
    return output


def _query_tier(
    query_embedding: list[float],
    tier: str,
    era: Optional[str],
    max_results: int,
) -> list[dict]:
    """Query ChromaDB for chunks filtered by tier (and optionally era).

    Parameters
    ----------
    query_embedding : list[float]
        Pre-computed embedding vector for the query.
    tier : str
        Tier to filter on ('book', 'media', 'forum', 'image').
    era : str | None
        If provided and not GENERAL, adds an era filter.
    max_results : int
        Maximum number of results to return from ChromaDB.

    Returns
    -------
    list[dict]
        Parsed results sorted by cosine distance (ascending).
    """
    collection = _get_collection()

    # Build where filter
    if era and era != "GENERAL":
        where_filter: dict = {"$and": [{"era": era}, {"tier": tier}]}
    else:
        where_filter = {"tier": tier}

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results,
            where=where_filter,  # type: ignore[arg-type]
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.warning("Tier query failed (tier=%s, era=%s): %s", tier, era, e)
        return []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    parsed = _parse_results(documents, metadatas, distances)
    # Sort by distance (ascending — closest first)
    parsed.sort(key=lambda r: r.get("metadata", {}).get("distance", 999.0))
    return parsed


def retrieve_context(
    query: str,
    era: Optional[Union[str, list[str]]] = None,
    top_k: int = 10,
) -> list[dict]:
    """Query ChromaDB for chunks relevant to *query* using tiered retrieval.

    Queries books first, then media, then forum.  Each tier gets a character
    budget (from config).  Unused budget flows downward: book → media → forum.

    When multiple eras are provided, book queries are split across each era
    with the book budget divided evenly.  Media and forum use no era filter
    for cross-era questions.

    Parameters
    ----------
    query : str
        The user's question or search text.
    era : str | list[str] | None
        If a single string, results are filtered to this era.
        If a list, books are queried per-era with split budget;
        media and forum use no era filter.
    top_k : int
        Maximum number of results to return (kept for backward compat but
        budget enforcement is the primary limiter).

    Returns
    -------
    list[dict]
        Each dict has keys: text, source, era, type, tier, metadata.
        Book chunks first, then media, then forum.
        Returns an empty list when nothing relevant is found.
    """
    try:
        embedder = _get_embedder()
        query_embedding = embedder.encode(query).tolist()

        # Normalise era to a list for uniform handling
        if era is None:
            eras: list[Optional[str]] = [None]
        elif isinstance(era, str):
            eras = [era]
        else:
            eras = list(era)

        is_cross_era = len(eras) > 1

        # 1. Query books — split across eras when multiple
        max_books_per_era = max(1, RAG_BOOK_MAX_RESULTS // len(eras))
        book_results: list[dict] = []
        for e in eras:
            book_results.extend(
                _query_tier(query_embedding, "book", e, max_books_per_era)
            )
        # Re-sort merged results by distance
        book_results.sort(key=lambda r: r.get("metadata", {}).get("distance", 999.0))

        # Media & forum: use single era when available, no filter for cross-era
        media_era = eras[0] if not is_cross_era else None
        forum_era = eras[0] if not is_cross_era else None
        media_results = _query_tier(query_embedding, "media", media_era, RAG_MEDIA_MAX_RESULTS)
        forum_results = _query_tier(query_embedding, "forum", forum_era, RAG_FORUM_MAX_RESULTS)

        # 2. Apply the Budget cascade (pure: split + downward redistribution).
        alloc = allocate_budget_cascade(
            book_results,
            media_results,
            forum_results,
            total_budget=RAG_TOTAL_CONTEXT_BUDGET,
            book_pct=RAG_BUDGET_BOOK_PCT,
            media_pct=RAG_BUDGET_MEDIA_PCT,
            forum_pct=RAG_BUDGET_FORUM_PCT,
        )

        # 3. Log tier stats
        logger.info(
            "Tiered retrieval (eras=%s) — book: %d chunks (%d/%d chars), "
            "media: %d chunks (%d/%d chars), "
            "forum: %d chunks (%d/%d chars)",
            eras,
            len(alloc.book.chunks), alloc.book.used, alloc.book.budget,
            len(alloc.media.chunks), alloc.media.used, alloc.media.budget,
            len(alloc.forum.chunks), alloc.forum.used, alloc.forum.budget,
        )

        # 4. Return flat list: books first, then media, then forum
        return alloc.chunks

    except Exception as e:
        logger.warning("RAG retrieval failed: %s", e)
        return []


_citation_logger = logging.getLogger("rag_citations")


def log_citation_tiers(response_text: str, rag_results: list[dict]) -> None:
    """Parse [N] citation references from the LLM response and log tier usage.

    Parameters
    ----------
    response_text : str
        The LLM-generated response text.
    rag_results : list[dict]
        The flat list of RAG results (1-indexed in the prompt).
    """
    citation_nums = re.findall(r"\[(\d+)\]", response_text)
    if not citation_nums:
        _citation_logger.info("Citation tier usage: no citations found in response")
        return

    tier_counts: dict[str, int] = {}
    for num_str in citation_nums:
        idx = int(num_str) - 1  # citations are 1-indexed
        if 0 <= idx < len(rag_results):
            tier = rag_results[idx].get("tier", "unknown")
        else:
            tier = "unknown"
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    parts = [f"{tier}={count}" for tier, count in sorted(tier_counts.items())]
    _citation_logger.info("Citation tier usage: %s", ", ".join(parts))
