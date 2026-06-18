"""
Played-passage tone feature extraction.

Sibling of the spit-buzz feature extractor; it *reuses the spit-buzz feature
library* (``audio.spit_buzz_features``) note-by-note and adds the two
melodic-tone measures the spit buzz doesn't need: how centered/in-tune each
note is (cents from equal temperament) and how cleanly each note is stopped
(residual air tail after the tongue stop).

This module needs librosa/numpy (it touches audio). The scoring side
(``audio.tone_scorer`` / ``audio.tone_calibration``) is kept pure so it can
be tested without the audio stack.

Per-note feature keys (the contract with the scorer):
    cents_abs               -- abs cents from nearest equal-tempered pitch
    spectral_centroid       -- brightness (Hz)
    high_freq_energy_ratio  -- energy above 4 kHz / total
    pitch_waver             -- std of f0 over the note (Hz); waver
    transient_energy_ratio  -- note-start pop strength
    attack_time             -- onset-to-peak time (s)
    release_air_tail        -- tail energy after the note / body energy
    pitch_midi              -- the note's MIDI pitch (None if unpitched); not a
                               tone dimension -- carried through the
                               scorer for the downstream per-note consumer

Python 3.9. ASCII-only.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np

from audio.spit_buzz_features import extract_spit_buzz_features
from audio import pitch_utils

logger = logging.getLogger(__name__)

DEFAULT_SR = 22050
HOP_LENGTH = 512
_MIN_NOTE_SEC = 0.12  # ignore segments shorter than this


def extract_played_passage_features(
    audio: Union[str, np.ndarray],
    sr: int = DEFAULT_SR,
) -> List[Dict[str, Any]]:
    """Segment a played passage into notes and extract per-note tone features.

    Args:
        audio: file path or numpy samples.
        sr: sample rate (used when *audio* is an ndarray; also the load
            target sr for a path).

    Returns:
        One feature dict per detected note, in time order. Empty list when
        no usable notes are found.
    """
    y, sr = _load_audio(audio, sr)
    if len(y) == 0:
        return []

    notes: List[Dict[str, Any]] = []
    for start, end in _segment_notes(y, sr):
        note_audio = y[start:end]
        if len(note_audio) < int(_MIN_NOTE_SEC * sr):
            continue
        feats = _extract_note_features(note_audio, sr)
        feats["_start_time"] = round(start / sr, 3)
        feats["_end_time"] = round(end / sr, 3)
        notes.append(feats)
    return notes


def _extract_note_features(note_audio: np.ndarray, sr: int) -> Dict[str, Any]:
    """Extract the tone feature set for a single note window."""
    # Reuse the spit-buzz feature library for the shared dimensions.
    sb = extract_spit_buzz_features(note_audio, sr=sr)

    feats: Dict[str, Any] = {
        "spectral_centroid": float(sb.get("spectral_centroid", 0.0)),
        "high_freq_energy_ratio": float(sb.get("high_freq_energy_ratio", 0.0)),
        # spit-buzz "pitch_stability" is the std of f0 -- exactly our waver.
        "pitch_waver": float(sb.get("pitch_stability", 0.0)),
        "transient_energy_ratio": float(sb.get("transient_energy_ratio", 0.0)),
        "attack_time": float(sb.get("attack_time", 0.0)),
    }
    median_f0 = _median_f0(note_audio, sr)
    feats["cents_abs"] = _cents_off_pitch(median_f0)
    feats["pitch_midi"] = _f0_to_midi(median_f0)
    feats["release_air_tail"] = _release_air_tail(note_audio, sr)
    return feats


def _median_f0(note_audio: np.ndarray, sr: int) -> Optional[float]:
    """Median voiced f0 (Hz) over the note, or ``None`` when unpitched."""
    try:
        f0, _, _ = librosa.pyin(
            note_audio,
            fmin=float(librosa.note_to_hz("E1")),
            fmax=float(librosa.note_to_hz("C7")),
            sr=sr,
            hop_length=HOP_LENGTH,
        )
        voiced = f0[~np.isnan(f0)]
        if len(voiced) == 0:
            return None
        median_f0 = float(np.median(voiced))
        return median_f0 if median_f0 > 0 else None
    except Exception:
        return None


def _cents_off_pitch(median_f0: Optional[float]) -> float:
    """Abs cents from the nearest equal-tempered pitch (lower == centered).

    A note with no detectable pitch is treated as 50 cents off -- the far end,
    so an unpitched/airy note never scores as centered.
    """
    if median_f0 is None:
        return 50.0
    _, cents = pitch_utils.freq_to_note_name(median_f0)
    return float(abs(cents))


def _f0_to_midi(median_f0: Optional[float]) -> Optional[int]:
    """MIDI pitch number nearest to *median_f0* (A4=440 -> 69), or ``None``.

    Carried through the scorer so the downstream per-note consumer can read a
    per-note pitch; not a tone feature itself.
    """
    if median_f0 is None or median_f0 <= 0:
        return None
    return int(round(69.0 + 12.0 * math.log2(median_f0 / 440.0)))


def _release_air_tail(note_audio: np.ndarray, sr: int) -> float:
    """Residual energy after the note relative to its body (lower == cleaner).

    A clean tongue-stop ending cuts the tone off sharply, leaving little
    energy in the final slice; an air-tailed ending bleeds breath noise.
    """
    n = len(note_audio)
    tail_samples = int(0.05 * sr)
    if n < tail_samples * 3:
        return 0.0
    body = note_audio[: n - tail_samples]
    tail = note_audio[n - tail_samples:]
    body_rms = float(np.sqrt(np.mean(body ** 2)))
    tail_rms = float(np.sqrt(np.mean(tail ** 2)))
    if body_rms < 1e-10:
        return 0.0
    return float(max(0.0, min(2.0, tail_rms / body_rms)))


def _segment_notes(y: np.ndarray, sr: int) -> List[Tuple[int, int]]:
    """Onset-based note segmentation. Returns (start, end) sample pairs.

    Falls back to a single whole-clip segment when no onsets are detected
    (mirrors the spit-buzz scorer's single-segment fallback).
    """
    try:
        onset_frames = librosa.onset.onset_detect(  # type: ignore[attr-defined]
            y=y, sr=sr, hop_length=HOP_LENGTH, backtrack=True,
        )
        onset_samples = librosa.frames_to_samples(onset_frames, hop_length=HOP_LENGTH)
    except Exception:
        onset_samples = np.array([], dtype=int)

    starts = [int(s) for s in onset_samples if 0 <= int(s) < len(y)]
    if not starts or starts[0] > int(0.05 * sr):
        starts = [0] + starts

    segments: List[Tuple[int, int]] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(y)
        if end - start > 0:
            segments.append((start, end))
    return segments


def _load_audio(audio: Union[str, np.ndarray], sr: int) -> Tuple[np.ndarray, int]:
    if isinstance(audio, str):
        y, sr_loaded = librosa.load(audio, sr=sr, mono=True)
        return y, int(sr_loaded)
    if isinstance(audio, np.ndarray):
        return audio, sr
    raise TypeError("Expected str path or np.ndarray, got %s" % type(audio))
