"""Percentile mapping shared by the audio scorers.

Single owner of "where does this feature value fall within a reference
distribution" -- the one move both the spit-buzz scorer (``audio.spit_buzz_scorer``)
and the played-passage tone scorer (``audio.tone_scorer``) make. Previously each
had its own copy: spit-buzz used ``np.searchsorted``, tone used ``bisect`` in
pure Python so it could be unit-tested without the audio stack -- two
implementations of one idea, free to drift at the edges.

This module is the pure-Python (bisect, no numpy/librosa) home for that idea,
mirroring how ``audio.spectral_features`` owns the low-level FFT math one layer
down. Bands and feedback thresholds are NOT shared -- those genuinely differ
between the two scorers -- only the value-to-percentile mapping is.

Python 3.9. ASCII-only.
"""

from __future__ import annotations

import bisect
from typing import Sequence


def value_to_percentile(
    value: float,
    sorted_refs: Sequence[float],
    *,
    inverted: bool = False,
) -> float:
    """Percentile of *value* within an ascending *sorted_refs* distribution.

    Returns the fraction of reference values at or below *value*, scaled to
    [0, 100] -- equivalent to ``np.searchsorted(side="right") / n * 100``.

    *sorted_refs* MUST be sorted ascending; the caller owns the reference
    distribution and both scorers store it sorted. An empty distribution
    returns the neutral 50.0.

    When *inverted* is True -- a feature where a LOWER raw value means BETTER
    quality (e.g. cents-off-pitch, attack time) -- the percentile is flipped to
    ``100 - pct`` so that a higher returned score always means better.
    """
    n = len(sorted_refs)
    if n == 0:
        return 50.0
    pos = bisect.bisect_right(sorted_refs, value)
    pct = max(0.0, min(100.0, (pos / n) * 100.0))
    return 100.0 - pct if inverted else pct
