"""
Spit buzz segment detector.

Detects segments of spit buzzing within audio that may contain talking,
silence, and demonstrations. Uses energy-based voice activity detection
followed by spectral classification to distinguish buzzing from speech.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union

import librosa
import numpy as np
from scipy.signal import find_peaks, peak_prominences

logger = logging.getLogger(__name__)

# Detection parameters
DEFAULT_SR = 22050
MIN_SEGMENT_DURATION = 0.3  # seconds
MERGE_GAP_THRESHOLD = 0.2  # seconds — merge same-type segments closer than this
HOP_LENGTH = 512
RMS_NOISE_FLOOR_MULTIPLIER = 3.0


class SpitBuzzDetector:
    """Detects and classifies spit buzz vs speech segments in audio."""

    def __init__(
        self,
        sr: int = DEFAULT_SR,
        hop_length: int = HOP_LENGTH,
        min_segment_duration: float = MIN_SEGMENT_DURATION,
        merge_gap: float = MERGE_GAP_THRESHOLD,
    ) -> None:
        self.sr = sr
        self.hop_length = hop_length
        self.min_segment_duration = min_segment_duration
        self.merge_gap = merge_gap

    def detect(
        self,
        audio: Union[str, np.ndarray],
        sr: int = DEFAULT_SR,
    ) -> List[Dict[str, Any]]:
        """
        Detect spit buzz and speech segments in audio.

        Args:
            audio: File path (str) or numpy array of audio samples.
            sr: Sample rate (used when *audio* is an ndarray).

        Returns:
            List of segment dicts, each with keys:
                start_time, end_time, confidence, segment_type ('buzz' or 'speech')
        """
        y, sr = self._load(audio, sr)

        # Phase 1: energy-based segmentation — find non-silence regions
        raw_segments = self._find_active_segments(y, sr)
        if not raw_segments:
            logger.info("No active segments found in audio.")
            return []

        # Phase 2: classify each active segment as buzz or speech
        classified = []
        for start_frame, end_frame in raw_segments:
            start_sample = start_frame * self.hop_length
            end_sample = min(end_frame * self.hop_length, len(y))
            segment_audio = y[start_sample:end_sample]

            if len(segment_audio) < self.sr * 0.05:
                continue  # too short to analyse

            seg_type, confidence = self._classify_segment(segment_audio, sr)
            start_time = float(start_frame * self.hop_length / sr)
            end_time = float(end_sample / sr)

            classified.append({
                "start_time": start_time,
                "end_time": end_time,
                "confidence": round(confidence, 3),
                "segment_type": seg_type,
            })

        # Post-processing
        merged = self._merge_adjacent(classified)
        filtered = [s for s in merged if (s["end_time"] - s["start_time"]) >= self.min_segment_duration]

        logger.info(
            "Detected %d segments (%d buzz, %d speech)",
            len(filtered),
            sum(1 for s in filtered if s["segment_type"] == "buzz"),
            sum(1 for s in filtered if s["segment_type"] == "speech"),
        )
        return filtered

    # ------------------------------------------------------------------
    # Phase 1 — energy-based segmentation
    # ------------------------------------------------------------------

    def _find_active_segments(
        self, y: np.ndarray, sr: int
    ) -> List[tuple]:
        """Return list of (start_frame, end_frame) tuples for non-silence regions."""
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]

        # Dynamic threshold: median RMS as noise floor estimate, scaled up
        noise_floor = float(np.median(rms))
        threshold = noise_floor * RMS_NOISE_FLOOR_MULTIPLIER
        # Ensure threshold is not absurdly low for very quiet recordings
        threshold = max(threshold, float(np.max(rms)) * 0.05)

        active = rms > threshold

        # Group consecutive active frames into segments
        segments: List[tuple] = []
        in_segment = False
        seg_start = 0

        for i, is_active in enumerate(active):
            if is_active and not in_segment:
                seg_start = i
                in_segment = True
            elif not is_active and in_segment:
                segments.append((seg_start, i))
                in_segment = False

        if in_segment:
            segments.append((seg_start, len(active)))

        return segments

    # ------------------------------------------------------------------
    # Phase 2 — spectral classification
    # ------------------------------------------------------------------

    def _classify_segment(
        self, segment: np.ndarray, sr: int
    ) -> tuple:
        """
        Classify an audio segment as 'buzz' or 'speech'.

        Distinguishing features:
        - Spectral centroid: buzzing tends higher
        - Spectral flatness: speech has more variation, buzzing is more tonal
        - Harmonic-to-noise ratio: buzzing is more harmonic
        - Temporal stability of spectral features: buzzing is more stable

        Returns:
            (segment_type, confidence) where confidence is 0-1.
        """
        n_fft = min(2048, len(segment))
        hop = self.hop_length

        # --- Feature extraction ---
        centroid = librosa.feature.spectral_centroid(
            y=segment, sr=sr, n_fft=n_fft, hop_length=hop
        )[0]
        flatness = librosa.feature.spectral_flatness(
            y=segment, n_fft=n_fft, hop_length=hop
        )[0]

        # Harmonic-to-noise ratio
        y_harm, y_perc = librosa.effects.hpss(segment)
        harm_energy = float(np.sum(y_harm ** 2))
        perc_energy = float(np.sum(y_perc ** 2))
        hnr = harm_energy / (perc_energy + 1e-10)

        # Temporal stability: std of frame-level centroid, normalised by mean
        centroid_mean = float(np.mean(centroid))
        centroid_std = float(np.std(centroid))
        centroid_cv = centroid_std / (centroid_mean + 1e-10)  # coefficient of variation

        flatness_mean = float(np.mean(flatness))
        flatness_std = float(np.std(flatness))

        # --- Scoring ---
        # Each feature contributes a buzz-likelihood score [0, 1]
        scores: List[float] = []

        # 1. Spectral centroid — buzzing tends to have higher centroid
        #    Speech centroid ~500-2000 Hz, buzzing often 1500-5000+
        centroid_score = _sigmoid(centroid_mean, midpoint=1800.0, steepness=0.002)
        scores.append(centroid_score)

        # 2. Spectral flatness — lower mean = more tonal = more buzz-like
        flatness_score = 1.0 - _sigmoid(flatness_mean, midpoint=0.15, steepness=30.0)
        scores.append(flatness_score)

        # 3. Harmonic-to-noise ratio — higher = more harmonic = more buzz-like
        hnr_score = _sigmoid(hnr, midpoint=3.0, steepness=0.5)
        scores.append(hnr_score)

        # 4. Temporal stability — lower centroid CV = more stable = more buzz-like
        stability_score = 1.0 - _sigmoid(centroid_cv, midpoint=0.4, steepness=5.0)
        scores.append(stability_score)

        # 5. Spectral flatness variation — speech has more flatness variation
        flatness_var_score = 1.0 - _sigmoid(flatness_std, midpoint=0.08, steepness=30.0)
        scores.append(flatness_var_score)

        # Weighted average
        weights = [0.20, 0.20, 0.25, 0.20, 0.15]
        buzz_likelihood = sum(s * w for s, w in zip(scores, weights))

        if buzz_likelihood >= 0.5:
            return "buzz", buzz_likelihood
        else:
            return "speech", 1.0 - buzz_likelihood

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _merge_adjacent(
        self, segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge adjacent segments of the same type when gap < merge_gap."""
        if not segments:
            return []

        merged: List[Dict[str, Any]] = [segments[0].copy()]

        for seg in segments[1:]:
            prev = merged[-1]
            gap = seg["start_time"] - prev["end_time"]

            if seg["segment_type"] == prev["segment_type"] and gap < self.merge_gap:
                # Merge: extend previous segment
                prev["end_time"] = seg["end_time"]
                prev["confidence"] = max(prev["confidence"], seg["confidence"])
            else:
                merged.append(seg.copy())

        return merged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load(
        self, audio: Union[str, np.ndarray], sr: int
    ) -> tuple:
        """Load audio from path or pass through ndarray."""
        if isinstance(audio, str):
            y, sr_loaded = librosa.load(audio, sr=self.sr, mono=True)
            return y, sr_loaded
        elif isinstance(audio, np.ndarray):
            return audio, sr
        else:
            raise TypeError(f"Expected str path or np.ndarray, got {type(audio)}")


def _sigmoid(x: float, midpoint: float = 0.0, steepness: float = 1.0) -> float:
    """Compute sigmoid mapping x → (0, 1) with configurable midpoint and steepness."""
    z = steepness * (x - midpoint)
    # Clamp to avoid overflow
    z = max(-20.0, min(20.0, z))
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------------------
# Attack-based repetition detection (standalone function)
# ------------------------------------------------------------------

DEFAULT_MIN_SEPARATION = 0.25  # seconds — default minimum gap between attacks


def detect_spitbuzz_attacks(
    y: np.ndarray,
    sr: int,
    expected_count: Optional[int] = None,
    expected_bpm: Optional[float] = None,
    session_id: Optional[str] = None,
    drill_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Detect individual spit-buzz attacks using RMS envelope peak finding.

    Unlike the segment-based SpitBuzzDetector, this function counts discrete
    attacks (pops) by looking for prominence peaks in the RMS energy envelope.

    Args:
        y: Audio samples (mono, float).
        sr: Sample rate.
        expected_count: Expected number of attacks (informational, not used for gating).
        expected_bpm: Expected tempo in BPM. When provided, the minimum separation
            between attacks is set to 0.6 * (60/bpm) seconds.
        session_id: Session ID for debug telemetry.
        drill_id: Drill ID for debug telemetry.

    Returns:
        List of attack dicts, each with keys:
            timestamp, prominence, peak_rms, window_start, window_end
    """
    hop_length = HOP_LENGTH

    # Compute RMS envelope
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # Determine minimum separation between peaks
    if expected_bpm is not None and expected_bpm > 0:
        min_separation = 0.6 * (60.0 / expected_bpm)
    else:
        min_separation = DEFAULT_MIN_SEPARATION

    # Convert min_separation from seconds to RMS frames
    min_separation_frames = int(min_separation * sr / hop_length)
    min_separation_frames = max(1, min_separation_frames)

    # Adaptive prominence threshold: 15% of max RMS
    rms_max = float(np.max(rms))
    prominence_threshold = 0.15 * rms_max

    # Find peaks with prominence gating and minimum separation
    peaks, properties = find_peaks(
        rms,
        distance=min_separation_frames,
        prominence=prominence_threshold,
    )

    # Build attack list with per-attack windows
    attacks: List[Dict[str, Any]] = []
    for i, peak_frame in enumerate(peaks):
        peak_sample = int(peak_frame) * hop_length
        timestamp = float(peak_sample) / sr
        peak_rms_val = float(rms[int(peak_frame)])
        prominence_val = float(properties["prominences"][i])

        # Attack window: [-80ms, +250ms] around peak
        window_start = max(0, peak_sample - int(0.08 * sr))
        window_end = min(len(y), peak_sample + int(0.25 * sr))

        attacks.append({
            "timestamp": round(timestamp, 4),
            "prominence": round(prominence_val, 6),
            "peak_rms": round(peak_rms_val, 6),
            "window_start": window_start,
            "window_end": window_end,
        })

    logger.info(
        "Attack detection: found %d attacks (expected=%s, bpm=%s, min_sep=%.3fs, prom_thresh=%.6f)",
        len(attacks),
        expected_count,
        expected_bpm,
        min_separation,
        prominence_threshold,
    )

    # Debug telemetry: structured JSON log with accepted and rejected peaks
    from app.core.config import DEBUG_SPITBUZZ
    if DEBUG_SPITBUZZ:
        _log_attack_debug(
            rms, sr, hop_length, y, peaks, prominence_threshold,
            min_separation_frames, min_separation, attacks,
            expected_count, expected_bpm, session_id, drill_id,
        )

    return attacks


def _log_attack_debug(
    rms: np.ndarray,
    sr: int,
    hop_length: int,
    y: np.ndarray,
    accepted_peaks: np.ndarray,
    prominence_threshold: float,
    min_separation_frames: int,
    min_separation: float,
    attacks: List[Dict[str, Any]],
    expected_count: Optional[int],
    expected_bpm: Optional[float],
    session_id: Optional[str],
    drill_id: Optional[str],
) -> None:
    """Log structured debug telemetry for attack detection."""
    accepted_set = set(int(p) for p in accepted_peaks)

    # Find ALL local maxima (no filtering) to identify rejected candidates
    all_peaks, _ = find_peaks(rms)
    all_prominences = peak_prominences(rms, all_peaks)[0]

    rejected: List[Dict[str, Any]] = []
    for i, peak_frame in enumerate(all_peaks):
        if int(peak_frame) in accepted_set:
            continue
        prom = float(all_prominences[i])
        timestamp = round(float(int(peak_frame) * hop_length) / sr, 4)

        if prom < prominence_threshold:
            reason = "low_prominence"
        else:
            # Passed prominence but was suppressed by distance constraint
            reason = "too_close"

        rejected.append({
            "timestamp": timestamp,
            "prominence": round(prom, 6),
            "reason": reason,
        })

    debug_data = {
        "session_id": session_id,
        "drill_id": drill_id,
        "expected_count": expected_count,
        "expected_bpm": expected_bpm,
        "detected_count": len(attacks),
        "attacks": [{"timestamp": a["timestamp"], "prominence": a["prominence"]} for a in attacks],
        "rejected_peaks": rejected,
        "min_separation_used": round(min_separation, 4),
        "prominence_threshold": round(prominence_threshold, 6),
        "audio_duration": round(float(len(y)) / sr, 3),
    }
    logger.info("spitbuzz_debug: %s", json.dumps(debug_data))
