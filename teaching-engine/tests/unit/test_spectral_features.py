"""Unit tests for the shared ``spectral_features`` module.

The spectral feature math is tested at the pure-function seam: synthetic
numpy arrays (pure sine, white noise, silence, decaying tone) go in,
scalar features come out, and we assert they fall in known ranges. No
audio files, no librosa.load.

Run from ``teaching-engine/``::

    python -m pytest tests/unit/test_spectral_features.py -v

Prior art: ``tests/unit/test_dp_tier_a.py`` (pure-function tests on arrays).
"""

from typing import Optional

import numpy as np
import pytest

from audio.spectral_features import (  # type: ignore[import]
    harmonic_to_noise_ratio,
    rms,
    rms_frames,
    spectral_centroid,
    spectral_centroid_frames,
    spectral_flatness,
    spectral_flatness_frames,
)


SR = 22050


# ---------------------------------------------------------------------------
# Synthetic signal builders
# ---------------------------------------------------------------------------

def _sine(freq_hz: float, duration_s: float = 1.5, amp: float = 0.3,
          sr: int = SR) -> np.ndarray:
    t = np.linspace(0.0, duration_s, int(duration_s * sr), endpoint=False)
    return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def _noise(duration_s: float = 1.5, amp: float = 0.3, sr: int = SR,
           seed: Optional[int] = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    return (amp * rng.standard_normal(n)).astype(np.float32)


def _silence(duration_s: float = 1.5, sr: int = SR) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def _decaying_tone(freq_hz: float = 440.0, duration_s: float = 1.5,
                   amp: float = 0.3, sr: int = SR) -> np.ndarray:
    t = np.linspace(0.0, duration_s, int(duration_s * sr), endpoint=False)
    env = np.exp(-3.0 * t)
    return (amp * env * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# spectral_centroid
# ---------------------------------------------------------------------------

def test_centroid_rises_with_higher_frequency():
    low = spectral_centroid(_sine(220.0), sr=SR)
    high = spectral_centroid(_sine(3000.0), sr=SR)
    assert high > low
    # A 220 Hz tone's centroid sits near the fundamental.
    assert 100.0 < low < 600.0


def test_centroid_frames_returns_array():
    frames = spectral_centroid_frames(_sine(440.0), sr=SR)
    assert isinstance(frames, np.ndarray)
    assert frames.ndim == 1
    assert frames.size > 1
    assert float(np.mean(frames)) == pytest.approx(
        spectral_centroid(_sine(440.0), sr=SR)
    )


# ---------------------------------------------------------------------------
# spectral_flatness
# ---------------------------------------------------------------------------

def test_flatness_near_one_for_noise():
    flat = spectral_flatness(_noise())
    assert 0.2 < flat <= 1.0


def test_flatness_near_zero_for_tone():
    flat = spectral_flatness(_sine(440.0))
    assert 0.0 <= flat < 0.05


def test_noise_is_flatter_than_tone():
    assert spectral_flatness(_noise()) > spectral_flatness(_sine(440.0))


# ---------------------------------------------------------------------------
# rms
# ---------------------------------------------------------------------------

def test_rms_scales_with_amplitude():
    quiet = rms(_sine(440.0, amp=0.1))
    loud = rms(_sine(440.0, amp=0.4))
    assert loud > quiet
    # RMS of a sine of amplitude a is a / sqrt(2).
    assert loud == pytest.approx(0.4 / np.sqrt(2), rel=0.1)


def test_rms_silence_is_zero():
    assert rms(_silence()) == pytest.approx(0.0, abs=1e-6)


def test_rms_frames_array_mean_matches_scalar():
    y = _sine(440.0)
    frames = rms_frames(y)
    assert isinstance(frames, np.ndarray)
    assert float(np.mean(frames)) == pytest.approx(rms(y))


# ---------------------------------------------------------------------------
# harmonic_to_noise_ratio  (canonical scale: raw harmonic / total energy, 0..1)
# ---------------------------------------------------------------------------

def test_hnr_high_for_clean_tone():
    hnr = harmonic_to_noise_ratio(_sine(440.0))
    assert 0.8 < hnr <= 1.0


def test_hnr_low_for_noise():
    hnr = harmonic_to_noise_ratio(_noise())
    assert hnr < 0.6


def test_hnr_tone_higher_than_noise():
    assert harmonic_to_noise_ratio(_sine(440.0)) > harmonic_to_noise_ratio(_noise())


def test_hnr_on_canonical_raw_ratio_scale():
    # Canonical scale is the raw harmonic/total-energy ratio, bounded in [0, 1]
    # — NOT a log10 value. A clean tone is close to 1, never above it.
    for y in (_sine(440.0), _noise(), _decaying_tone()):
        hnr = harmonic_to_noise_ratio(y)
        assert 0.0 <= hnr <= 1.0


def test_hnr_silence_does_not_crash():
    # Degenerate input must return a finite value, not NaN / div-by-zero.
    hnr = harmonic_to_noise_ratio(_silence())
    assert np.isfinite(hnr)
    assert 0.0 <= hnr <= 1.0
