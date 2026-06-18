"""Unified schema for double-pedal (DP) detection results.

This module is the single source of truth for what
``services.audio.dp_detector_router`` returns. Both the heuristic
detector path and the vNext (ML) detector path normalize to this shape so
``app/api/routes/lesson.py``, the SPA, and any future detector consume
the same dictionary keys.

Tier semantics
--------------
- Tier A (register gate, beginner): "is the student playing low enough?"
  No rung naming, no cents-vs-target tuning feedback. The Tier-B-only
  fields (``note_written``, ``note_concert``, ``cents_offset``,
  ``target_rung``, ``ambiguous``, ``runner_up``) are ``None`` for
  Tier A results.
- Tier B (advancing): adds rung identification on the 7-rung Yoga ladder
  plus signed cents offset against the chosen rung. All Tier-B-only
  fields are populated.
- Tier B downgraded to Tier A: when the drill or signal-quality gate
  forces a downgrade, ``tier == 'A'``, ``in_register`` reflects the
  Tier A gate result, ``tier_downgrade_reason`` names the cause, and
  the Tier-B-only fields stay ``None``. The downgrade taxonomy is owned
  by ``dp_vnext_mapping`` and the router; this schema only enforces the
  shape.

Field conventions
-----------------
- All Tier-B-only fields are ``Optional`` so the same ``TypedDict``
  represents both tiers. Tier A instances simply have ``None`` for those
  fields.
- ``confidence`` is bounded ``[0.0, 1.0]``.
- ``cents_offset`` is signed: negative = flat, positive = sharp,
  relative to the chosen rung.
- ``runner_up`` is populated only when ``ambiguous`` is ``True``; it
  carries the second-best rung candidate so the UI can disambiguate.
- ``register_status`` refines *why* a Tier A result is or is not in
  register, so the lesson can give the right coaching:
  ``'in_register'`` (good), ``'too_high_single_pedal'`` (a clear, stable
  tone but in the single-pedal register -- the student needs to get
  LOWER), ``'too_high_unstable'`` (pitch is too high AND the tone is not
  settling -- get lower and steadier), ``'not_tonal'`` (no stable buzz:
  noise / airy / silence), or
  ``'out_of_register'`` (the pure-rule fallback's generic miss). It is
  ``Optional`` and may be ``None`` on detector paths that do not compute
  it (e.g. vNext); consumers must treat ``None`` as "no extra detail".

Examples
--------
The three example dicts below are the canonical examples; they are
reproduced (verbatim) in ``docs/DP_DETECTOR_V2.md`` for reference.

Tier A result (beginner / register gate only)::

    {
        'tier': 'A',
        'in_register': True,
        'confidence': 0.82,
        'tier_downgrade_reason': None,
        'note_written': None,
        'note_concert': None,
        'cents_offset': None,
        'target_rung': None,
        'ambiguous': None,
        'runner_up': None,
    }

Tier B result (advancing student, all rung/cents fields populated)::

    {
        'tier': 'B',
        'in_register': True,
        'confidence': 0.91,
        'tier_downgrade_reason': None,
        'note_written': 'C2',
        'note_concert': 'Bb1',
        'cents_offset': -12.4,
        'target_rung': 'C2',
        'ambiguous': False,
        'runner_up': None,
    }

Tier B downgraded to Tier A (signal-quality block)::

    {
        'tier': 'A',
        'in_register': False,
        'confidence': 0.41,
        'tier_downgrade_reason': 'voiced_frame_ratio',
        'note_written': None,
        'note_concert': None,
        'cents_offset': None,
        'target_rung': None,
        'ambiguous': None,
        'runner_up': None,
    }
"""

from typing import Any, Dict, Literal, Optional

try:
    from typing import TypedDict
except ImportError:  # pragma: no cover - py<3.8 fallback (project is 3.9)
    from typing_extensions import TypedDict  # type: ignore[assignment]


class DPDetectionResult(TypedDict):
    """Unified return shape for the DP detector router.

    See module docstring for tier semantics and the canonical example
    dicts. All Tier-B-only fields use ``Optional`` so a Tier A result is
    representable in the same TypedDict by setting them to ``None``.
    """

    tier: Literal['A', 'B']
    in_register: bool
    confidence: float
    tier_downgrade_reason: Optional[str]
    register_status: Optional[str]
    note_written: Optional[str]
    note_concert: Optional[str]
    cents_offset: Optional[float]
    target_rung: Optional[str]
    ambiguous: Optional[bool]
    runner_up: Optional[Dict[str, Any]]


__all__ = ['DPDetectionResult']
