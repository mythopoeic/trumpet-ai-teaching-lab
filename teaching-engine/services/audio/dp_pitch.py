"""Subharmonic-summation F0 estimator for the double-pedal register.

Phone and laptop mics roll off the 41-58 Hz double-pedal fundamental, so pyin
(which hunts the fundamental's periodicity) returns NaN or garbage on them. But
the 2nd-12th harmonics (80-700 Hz) survive, and double-pedal rungs are fully
recoverable from those. ``subharmonic_f0`` scores each candidate F0 by the
spectral energy at its harmonics -- **skipping the fundamental** -- and returns
the best candidate. Validated cross-mic (USB / phone / laptop) on clean
sustained holds, where it tracks pyin within ~1 Hz and is in fact MORE robust
(it recovers rungs where pyin returns NaN).

This is the pitch estimator for Tier B note identification on all mics. It is a
separate path from the shared pyin extractor (it is not pyin, so it does not
violate the single-pyin-path invariant).

Public API
----------
- ``subharmonic_f0(audio, sample_rate, ...)`` -> (f0_hz, strength). ``f0_hz`` is
  NaN when the clip is too short / silent. ``strength`` is the normalized peak
  score in [0, 1] (a soft confidence; low means no clear pitch).

Python 3.9. ASCII-only.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# Candidate F0 search band: covers Gb (~44 Hz) to C (~60 Hz, plus the player's
# upward intonation) with margin. Span < 2x so there is no octave ambiguity.
FMIN_HZ = 38.0
FMAX_HZ = 67.0
STEP_HZ = 0.1
# Harmonics summed. k starts at 2: the fundamental is unreliable / absent on
# phone+laptop, and including it pulls candidates toward low-frequency rumble.
K_MIN = 2
K_MAX = 12
# Analysis region: trim onset/offset so the estimate reflects the steady tone.
EDGE_TRIM_S = 0.05
MIN_SAMPLES = 2048


def subharmonic_f0(
    audio: np.ndarray,
    sample_rate: int,
    fmin: float = FMIN_HZ,
    fmax: float = FMAX_HZ,
    step: float = STEP_HZ,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
) -> Tuple[float, float]:
    """Estimate double-pedal F0 by subharmonic summation. Returns (f0, strength).

    For each candidate F0 in ``[fmin, fmax)``, sum the (linearly interpolated)
    magnitude spectrum at harmonics ``k_min..k_max``, weighted ``1/k``. The
    argmax is the F0. ``strength`` normalizes the winning score by the mean
    score across candidates, mapped to ~[0, 1] -- a soft confidence that is low
    when no candidate stands out (noise / silence).
    """
    arr = np.asarray(audio, dtype=np.float64).reshape(-1)
    if arr.size < MIN_SAMPLES or sample_rate <= 0:
        return float('nan'), 0.0

    trim = int(EDGE_TRIM_S * sample_rate)
    if arr.size > 3 * trim:
        arr = arr[trim:arr.size - trim]
    n = arr.size
    if n < MIN_SAMPLES:
        return float('nan'), 0.0

    if float(np.dot(arr, arr)) <= 0.0:
        return float('nan'), 0.0

    nfft = 1 << int(math.ceil(math.log2(n * 2)))
    mag = np.abs(np.fft.rfft(arr * np.hanning(n), nfft))
    bin_hz = float(sample_rate) / float(nfft)
    n_bins = mag.shape[0]

    candidates = np.arange(fmin, fmax, step)
    scores = np.empty(candidates.shape[0], dtype=np.float64)
    for ci, f0 in enumerate(candidates):
        s = 0.0
        for k in range(k_min, k_max + 1):
            idx = (k * f0) / bin_hz
            i0 = int(idx)
            if i0 + 1 >= n_bins:
                break
            frac = idx - i0
            s += (mag[i0] * (1.0 - frac) + mag[i0 + 1] * frac) / k
        scores[ci] = s

    if scores.size == 0 or float(scores.max()) <= 0.0:
        return float('nan'), 0.0
    best_i = int(scores.argmax())
    best_f0 = float(candidates[best_i])
    mean_score = float(scores.mean())
    peak = float(scores[best_i])
    # Strength: how much the peak exceeds the average candidate. 1.0 = peak is
    # 2x the mean or more; 0 = flat (no pitch).
    strength = 0.0 if mean_score <= 0.0 else max(0.0, min(1.0, (peak / mean_score - 1.0)))
    return best_f0, strength
