"""Shared spectral-feature definitions.

This module is the single owner of the low-level spectral features used
across the audio stack — spectral centroid, spectral flatness, RMS energy,
and harmonic-to-noise ratio. Each feature is defined exactly once here so
the general audio analyzer (:mod:`rag.audio_analyzer`) and the spit-buzz
feature extractor (:mod:`audio.spit_buzz_features`) cannot drift in how
they compute it.

Design notes
------------
* **Pure functions.** Every function takes a numpy waveform (and optional
  frame parameters) and returns a value. No file I/O, no librosa.load, no
  hidden state — so they can be exercised on synthetic signals in tests.
* **One scale per feature.** In particular ``harmonic_to_noise_ratio``
  returns the *raw harmonic / total-energy ratio* in ``[0, 1]`` — never a
  log10 value. Historically the general analyzer reported a log10-scaled
  HNR under the same key as the spit-buzz extractor's raw ratio, and a
  downstream consumer assumed the log scale: a silent unit mismatch. The
  raw ratio is now the one canonical scale. If a consumer ever needs a
  decibel form, add a clearly named ``*_db`` helper rather than overloading
  this one.
* **Scope.** The double-pedal detector (see ``docs/DP_DETECTOR_V2.md``,
  FR-8) deliberately keeps its own isolated, frozen-threshold feature
  extraction and is intentionally *not* folded into this module.

Frame parameters default to librosa's conventions (``n_fft=2048``,
``hop_length=512``, ``frame_length=2048``). Callers that need different
framing pass it explicitly; the computation logic still lives here.
"""

from __future__ import annotations

import librosa
import numpy as np

__all__ = [
    "DEFAULT_N_FFT",
    "DEFAULT_HOP_LENGTH",
    "DEFAULT_FRAME_LENGTH",
    "spectral_centroid",
    "spectral_centroid_frames",
    "spectral_flatness",
    "spectral_flatness_frames",
    "rms",
    "rms_frames",
    "harmonic_to_noise_ratio",
]

DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_FRAME_LENGTH = 2048

# Guards a division so degenerate (silent) input yields 0.0 rather than NaN.
_EPS = 1e-10


# ---------------------------------------------------------------------------
# Spectral centroid (brightness, Hz)
# ---------------------------------------------------------------------------

def spectral_centroid_frames(
    y: np.ndarray,
    sr: int,
    *,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> np.ndarray:
    """Per-frame spectral centroid in Hz.

    Returns the 1-D array of per-frame centroids so callers that need a
    spread (std/max/min) can aggregate it themselves.
    """
    return librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]


def spectral_centroid(
    y: np.ndarray,
    sr: int,
    *,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> float:
    """Mean spectral centroid in Hz (brightness / timbre).

    Higher for brighter, higher-frequency content. Bounded by ``[0, sr/2]``.
    """
    return float(np.mean(spectral_centroid_frames(
        y, sr, n_fft=n_fft, hop_length=hop_length
    )))


# ---------------------------------------------------------------------------
# Spectral flatness (tonal vs noise-like, 0..1)
# ---------------------------------------------------------------------------

def spectral_flatness_frames(
    y: np.ndarray,
    *,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> np.ndarray:
    """Per-frame spectral flatness, each value in ``[0, 1]``."""
    return librosa.feature.spectral_flatness(
        y=y, n_fft=n_fft, hop_length=hop_length
    )[0]


def spectral_flatness(
    y: np.ndarray,
    *,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> float:
    """Mean spectral flatness in ``[0, 1]``.

    Near ``0`` for a pure tone (energy concentrated in harmonics), near
    ``1`` for white noise (energy spread flat across the spectrum).
    """
    return float(np.mean(spectral_flatness_frames(
        y, n_fft=n_fft, hop_length=hop_length
    )))


# ---------------------------------------------------------------------------
# RMS energy (dynamics)
# ---------------------------------------------------------------------------

def rms_frames(
    y: np.ndarray,
    *,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> np.ndarray:
    """Per-frame root-mean-square energy.

    Returns the 1-D array of per-frame RMS values so callers can compute
    mean/std/max/min for a dynamics summary.
    """
    return librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length
    )[0]


def rms(
    y: np.ndarray,
    *,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> float:
    """Mean RMS energy. Scales with amplitude; ``0`` for silence."""
    return float(np.mean(rms_frames(
        y, frame_length=frame_length, hop_length=hop_length
    )))


# ---------------------------------------------------------------------------
# Harmonic-to-noise ratio (canonical scale)
# ---------------------------------------------------------------------------

def harmonic_to_noise_ratio(y: np.ndarray) -> float:
    """Harmonic-to-noise ratio on the **one canonical scale**.

    Defined as the raw ratio of harmonic energy to total energy::

        harmonic_energy / total_energy

    using librosa's harmonic/percussive source separation. The result is a
    raw ratio in ``[0, 1]`` — **not** a log10 value:

    * close to ``1`` for a clean, tonal signal (most energy is harmonic),
    * low for broadband noise (harmonic energy is a small fraction).

    This is the single documented scale; do not reintroduce a log10 form
    under this name. Silent input returns ``0.0`` rather than NaN.
    """
    total_energy = float(np.sum(y.astype(np.float64) ** 2))
    if total_energy <= _EPS:
        return 0.0
    y_harmonic, _ = librosa.effects.hpss(y)  # type: ignore[attr-defined]
    harmonic_energy = float(np.sum(y_harmonic.astype(np.float64) ** 2))
    ratio = harmonic_energy / (total_energy + _EPS)
    # hpss is not perfectly energy-preserving; clamp to the documented range.
    return float(min(1.0, max(0.0, ratio)))
