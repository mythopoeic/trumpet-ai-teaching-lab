"""Era detection via manual override, LLM classification, or keyword fallback.

Two routing schemes live here:

* The legacy ``detect_era`` precedence (override → LLM → keyword), still used by
  the Auto-mode pipeline.
* The ADR-0001 default-TCE scheme — ``resolve_method_era`` — which routes to the
  selected teaching method (TCE by default) and only leaves it when the student
  *names* another method, via the expansive name-only ``keyword_classify_era``.
  This path never calls the LLM classifier. See ``docs/adr/0001-*`` and the
  ``CONTEXT.md`` "Era disclosure" term.
"""

import logging
import re
from collections import Counter
from typing import List, Optional

logger = logging.getLogger(__name__)

_VALID_ERAS = {"TRUMPET_YOGA", "SUPERCHOPS", "TCE", "GENERAL"}

# ADR-0001: the era the bot defaults to when no method is named or selected.
_DEFAULT_ERA = "TCE"

_CLASSIFICATION_PROMPT = (
    "You are a classifier for Jerome Callet's trumpet teaching methods. "
    "Given a student's question, respond with EXACTLY one word — the era it belongs to:\n"
    "- TRUMPET_YOGA: questions about Trumpet Yoga method, breathing yoga, relaxation, natural embouchure\n"
    "- SUPERCHOPS: questions about Superchops method, forward lip position, compression, power playing\n"
    "- TCE: questions about Tongue Controlled Embouchure, tongue placement, tongue-controlled techniques\n"
    "- GENERAL: questions that don't clearly belong to one era, or are about trumpet playing in general\n\n"
    "Respond with only the era name, nothing else."
)


def detect_era(user_text: str, era_override: Optional[str] = None) -> str:
    """Detect which Callet teaching era a question relates to.

    Priority:
    1. Manual override (if provided and valid)
    2. LLM classification
    3. Keyword fallback

    Returns one of: TRUMPET_YOGA, SUPERCHOPS, TCE, GENERAL.
    """
    # 1. Manual override
    if era_override:
        normalized = _normalize_method(era_override)
        if normalized:
            return normalized
        logger.warning("Invalid era override '%s', falling back to detection", era_override)

    # 2. LLM classification
    try:
        result = _llm_classify_era(user_text)
        if result:
            return result
    except Exception:
        logger.warning("LLM era classification failed, falling back to keywords", exc_info=True)

    # 3. Keyword fallback
    return keyword_classify_era(user_text) or "GENERAL"


def _llm_classify_era(user_text: str) -> Optional[str]:
    """Use the LLM service to classify the era. Returns None on failure."""
    from app.services.llm import generate_response

    response = generate_response(
        system_prompt=_CLASSIFICATION_PROMPT,
        user_text=user_text,
    )

    # Parse the response — expect a single era name
    cleaned = response.strip().upper().replace(" ", "_").replace("-", "_")
    # Strip common mock prefixes like "[MOCK TRUMPET_YOGA]"
    if cleaned.startswith("["):
        # Extract era from mock format like "[MOCK TRUMPET_YOGA] ..."
        for era in _VALID_ERAS:
            if era in cleaned:
                return era
        return None

    if cleaned in _VALID_ERAS:
        return cleaned

    # Check if response contains a valid era (LLM might add extra text)
    for era in _VALID_ERAS:
        if era in cleaned:
            return era

    logger.warning("LLM returned unrecognized era: '%s'", response.strip())
    return None


# The trumpet model "Callet Superchops .464" (and the .460/.462 bore variants)
# is a horn, not the method — blank it before matching so it can't hit
# SUPERCHOPS. The book "Superchops" (no model number) is still allowed through.
_TRUMPET_MODEL_RE = re.compile(r"callet\s+superchops\s*\.?\d{3}", re.IGNORECASE)

# Name/title keyword tables, checked in order. TCE is listed before SUPERCHOPS
# so "Master Superchops" / "MSC" resolves to TCE before the bare "superchops"
# token can claim it. These are method/book *names* and abbreviations only —
# never technique terms — and use word boundaries so bare "yoga" / "SC" / "TS"
# can't leak an override.
_ERA_KEYWORDS = [
    ("TCE", [
        r"\btce\b",
        r"tongue[ -]controlled embouchure",
        r"trumpet secrets",
        r"master[ -]superchops",
        r"\bmsc\b",
        r"tongue chops",
        r"spit buzz",
    ]),
    ("SUPERCHOPS", [
        r"superchops",
        r"super chops",
        r"beyond arban",
    ]),
    ("TRUMPET_YOGA", [
        r"trumpet yoga",
        r"\bty\b",
        r"brass power and endurance",
    ]),
]
_ERA_PATTERNS = [
    (era, [re.compile(p, re.IGNORECASE) for p in patterns])
    for era, patterns in _ERA_KEYWORDS
]

# Pronoun-style follow-up: an immediate next turn with no new method name that
# references the prior answer ("tell me more about that"). Only these inherit
# the prior turn's era; any other un-named turn reverts to the default method.
_FOLLOWUP_PRONOUN_RE = re.compile(
    r"\b(that|it|this|those|these|them|they|he|him|his)\b", re.IGNORECASE
)


def keyword_classify_era(user_text: str) -> Optional[str]:
    """Detect an era from an explicit method/book *name* in the text.

    Name-only, expansive classification (ADR-0001): returns an era only when the
    query names a method, book, or recognized abbreviation — never on technique
    terms. Returns ``None`` when nothing is named, so the caller keeps the
    default/selected era. Trap handling: "Master Superchops"/"MSC" → TCE (before
    bare "superchops"), bare "yoga" never matches, and the "Callet Superchops
    .464" trumpet model does not route to SUPERCHOPS.
    """
    text = _TRUMPET_MODEL_RE.sub(" ", user_text)
    for era, patterns in _ERA_PATTERNS:
        if any(pattern.search(text) for pattern in patterns):
            return era
    return None


def _normalize_method(method: Optional[str]) -> Optional[str]:
    """Normalize a selected-method / prior-era value to a valid era, or None."""
    if not method:
        return None
    normalized = method.upper().replace(" ", "_").replace("-", "_")
    return normalized if normalized in _VALID_ERAS else None


def resolve_method_era(
    text: str,
    selected_method: Optional[str] = None,
    prior_era: Optional[str] = None,
    mode: str = "chat",
) -> str:
    """Resolve the era for one turn under ADR-0001 default-TCE routing.

    The resolution contract: ``resolve(text, selected_method, prior_era, mode)
    -> era``. Precedence:

    1. An explicit method **name** in ``text`` overrides everything (one-shot).
    2. Otherwise, a pronoun-style follow-up (no new name) inherits ``prior_era``
       — so "tell me more about that" stays in the just-named method.
    3. Otherwise, route to ``selected_method`` (TCE by default).

    The LLM era classifier is never consulted here. ``mode`` is part of the
    contract for parity with the lesson/chat seam but does not change routing in
    this default path.
    """
    named = keyword_classify_era(text)
    if named:
        return named

    if prior_era and _FOLLOWUP_PRONOUN_RE.search(text):
        inherited = _normalize_method(prior_era)
        if inherited:
            return inherited

    return _normalize_method(selected_method) or _DEFAULT_ERA


def determine_dominant_era(rag_results: List[dict]) -> str:
    """Determine the dominant era from RAG results.

    Counts the 'era' field across results, ignoring 'GENERAL'.
    On tie, returns the era of the highest-ranked (first) non-GENERAL result.
    Returns 'GENERAL' if no results or all results are GENERAL.
    """
    if not rag_results:
        return "GENERAL"

    # Count non-GENERAL eras
    era_counts = Counter(
        r["era"] for r in rag_results
        if r.get("era") and r["era"] != "GENERAL"
    )

    if not era_counts:
        return "GENERAL"

    max_count = max(era_counts.values())
    top_eras = [era for era, count in era_counts.items() if count == max_count]

    if len(top_eras) == 1:
        return top_eras[0]

    # Tie-break: return the era of the first non-GENERAL result
    for r in rag_results:
        era = r.get("era")
        if era and era != "GENERAL" and era in top_eras:
            return era

    return "GENERAL"
