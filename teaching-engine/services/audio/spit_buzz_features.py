"""
Spit buzz quality feature extraction.

Extracts measurable quality dimensions from spit buzz audio clips:
- Pop clarity: onset sharpness, transient energy, onset spectral spread
- Brilliance: spectral centroid, high-frequency energy ratio, spectral rolloff
- Sustain quality: energy stability, pitch stability
- Overblowing detection: spectral flatness, HNR, dynamic range compression
- Harmonic richness: number of harmonics, harmonic decay rate
"""

import logging
from typing import Dict, Any, List, Optional, Union

import librosa
import numpy as np

from audio import spectral_features as spec

logger = logging.getLogger(__name__)

DEFAULT_SR = 22050
HOP_LENGTH = 512


class SpitBuzzFeatures:
    """Container for all extracted spit buzz quality features."""

    def __init__(self, features: Dict[str, Any]) -> None:
        self._features = features

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._features)

    def __getitem__(self, key: str) -> Any:
        return self._features[key]

    def __repr__(self) -> str:
        return f"SpitBuzzFeatures({self._features})"


def extract_spit_buzz_features(
    audio: Union[str, np.ndarray],
    sr: int = DEFAULT_SR,
) -> Dict[str, Any]:
    """
    Extract all spit buzz quality features from an audio clip.

    Args:
        audio: File path (str) or numpy array of audio samples.
        sr: Sample rate (used when *audio* is an ndarray, or as target sr for file load).

    Returns:
        Dict with keys for each quality dimension and sub-features.
    """
    y, sr = _load_audio(audio, sr)

    if len(y) == 0:
        return _empty_features()

    features: Dict[str, Any] = {}

    # Compute shared spectral representations once
    n_fft = min(2048, len(y))
    stft = librosa.stft(y, n_fft=n_fft, hop_length=HOP_LENGTH)
    magnitude = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # -- 1. Pop Clarity --
    features.update(_extract_pop_clarity(y, sr, n_fft, magnitude, freqs))

    # -- 2. Brilliance --
    features.update(_extract_brilliance(y, sr, n_fft, magnitude, freqs))

    # -- 3. Sustain Quality --
    features.update(_extract_sustain_quality(y, sr, n_fft))

    # -- 4. Overblowing Detection --
    features.update(_extract_overblowing(y, sr, n_fft, magnitude))

    # -- 5. Harmonic Richness --
    features.update(_extract_harmonic_richness(y, sr, magnitude, freqs))

    return features


def extract_per_attack_features(
    y: np.ndarray,
    sr: int,
    attacks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract per-attack features from detected attack windows.

    For each attack window, extracts: pop_strength, spectral_centroid,
    spectral_flatness, hnr_proxy, decay_rate. Then computes aggregate
    stats (mean/std) and an overall attack_consistency score.

    Args:
        y: Audio samples as numpy array.
        sr: Sample rate.
        attacks: List of attack dicts from detect_spitbuzz_attacks(),
                 each with window_start and window_end (sample indices).

    Returns:
        Dict with 'per_attack' (List[Dict]), 'aggregate' (Dict), and
        'attack_consistency' (float).
    """
    if not attacks:
        return {
            "per_attack": [],
            "aggregate": {},
            "attack_consistency": 0.0,
        }

    per_attack: List[Dict[str, Any]] = []
    feature_names = [
        "pop_strength", "spectral_centroid", "spectral_flatness",
        "hnr_proxy", "decay_rate",
    ]

    for attack in attacks:
        window_start = int(attack["window_start"])
        window_end = int(attack["window_end"])
        window = y[window_start:window_end]

        if len(window) < HOP_LENGTH:
            per_attack.append({name: 0.0 for name in feature_names})
            continue

        n_fft = min(2048, len(window))
        stft = librosa.stft(window, n_fft=n_fft, hop_length=HOP_LENGTH)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # pop_strength via transient_energy_ratio
        pop_feats = _extract_pop_clarity(window, sr, n_fft, magnitude, freqs)
        pop_strength = float(pop_feats["transient_energy_ratio"])

        # spectral_centroid
        brilliance_feats = _extract_brilliance(window, sr, n_fft, magnitude, freqs)
        spectral_centroid = float(brilliance_feats["spectral_centroid"])

        # spectral_flatness and hnr_proxy
        overblow_feats = _extract_overblowing(window, sr, n_fft, magnitude)
        spectral_flatness = float(overblow_feats["spectral_flatness"])
        hnr_proxy = float(overblow_feats["harmonic_to_noise_ratio"])

        # decay_rate: RMS in two 50ms sub-windows after peak
        peak_sample = int(attack.get("peak_sample", window_start + int(0.08 * sr)))
        # Convert to window-relative offset
        peak_offset = peak_sample - window_start
        sub_window_samples = int(0.05 * sr)

        sub1_start = peak_offset
        sub1_end = min(sub1_start + sub_window_samples, len(window))
        sub2_start = sub1_end
        sub2_end = min(sub2_start + sub_window_samples, len(window))

        sub1 = window[sub1_start:sub1_end]
        sub2 = window[sub2_start:sub2_end]

        rms1 = float(np.sqrt(np.mean(sub1 ** 2))) if len(sub1) > 0 else 0.0
        rms2 = float(np.sqrt(np.mean(sub2 ** 2))) if len(sub2) > 0 else 0.0

        if rms1 > 1e-10:
            decay_rate = float((rms1 - rms2) / rms1)
        else:
            decay_rate = 0.0

        per_attack.append({
            "pop_strength": pop_strength,
            "spectral_centroid": spectral_centroid,
            "spectral_flatness": spectral_flatness,
            "hnr_proxy": hnr_proxy,
            "decay_rate": decay_rate,
        })

    # Aggregate stats: mean and std for each feature
    aggregate: Dict[str, Any] = {}
    cvs: List[float] = []

    for name in feature_names:
        values = np.array([a[name] for a in per_attack], dtype=np.float64)
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        aggregate[f"mean_{name}"] = mean_val
        aggregate[f"std_{name}"] = std_val

        # Coefficient of variation for consistency
        if abs(mean_val) > 1e-10:
            cvs.append(std_val / abs(mean_val))

    # attack_consistency: 1 - mean(CVs), clamped to [0, 1]
    if cvs:
        mean_cv = float(np.mean(np.array(cvs)))
        attack_consistency = max(0.0, min(1.0, 1.0 - mean_cv))
    else:
        attack_consistency = 0.0

    aggregate["attack_consistency"] = attack_consistency

    return {
        "per_attack": per_attack,
        "aggregate": aggregate,
        "attack_consistency": attack_consistency,
    }


# ------------------------------------------------------------------
# 1. Pop Clarity
# ------------------------------------------------------------------

def _extract_pop_clarity(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    magnitude: np.ndarray,
    freqs: np.ndarray,
) -> Dict[str, Any]:
    """
    Pop clarity features measure how clean and sharp the initial onset is.

    - attack_time: seconds from onset to peak energy in the first 50 ms
    - transient_energy_ratio: energy in the first 50 ms vs total energy
    - onset_spectral_spread: spectral bandwidth at the onset region
    """
    onset_samples = int(0.05 * sr)  # 50 ms window

    # Attack time: time from start to peak energy in onset window
    onset_region = y[: min(onset_samples, len(y))]
    if len(onset_region) == 0:
        return {
            "attack_time": 0.0,
            "transient_energy_ratio": 0.0,
            "onset_spectral_spread": 0.0,
        }

    rms_onset = spec.rms_frames(
        onset_region, frame_length=min(512, len(onset_region)), hop_length=128
    )

    if len(rms_onset) == 0:
        attack_time = 0.0
    else:
        peak_frame = int(np.argmax(rms_onset))
        attack_time = float(peak_frame * 128 / sr)

    # Transient energy ratio
    onset_energy = float(np.sum(onset_region ** 2))
    total_energy = float(np.sum(y ** 2))
    transient_energy_ratio = onset_energy / (total_energy + 1e-10)

    # Onset spectral spread (bandwidth at onset)
    onset_frames = max(1, onset_samples // HOP_LENGTH)
    if magnitude.shape[1] >= onset_frames:
        onset_magnitude = magnitude[:, :onset_frames]
    else:
        onset_magnitude = magnitude

    # Spectral bandwidth of the onset region
    onset_spectrum = np.mean(onset_magnitude, axis=1)
    total_power = float(np.sum(onset_spectrum ** 2))
    if total_power > 1e-10:
        centroid = float(np.sum(freqs * onset_spectrum ** 2) / total_power)
        onset_spectral_spread = float(
            np.sqrt(np.sum(((freqs - centroid) ** 2) * onset_spectrum ** 2) / total_power)
        )
    else:
        onset_spectral_spread = 0.0

    return {
        "attack_time": attack_time,
        "transient_energy_ratio": transient_energy_ratio,
        "onset_spectral_spread": onset_spectral_spread,
    }


# ------------------------------------------------------------------
# 2. Brilliance
# ------------------------------------------------------------------

def _extract_brilliance(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    magnitude: np.ndarray,
    freqs: np.ndarray,
) -> Dict[str, Any]:
    """
    Brilliance features measure brightness and high-frequency content.

    - spectral_centroid: mean spectral centroid (Hz)
    - high_freq_energy_ratio: energy above 4 kHz / total energy
    - spectral_rolloff: mean spectral rolloff frequency (Hz)
    """
    spectral_centroid = spec.spectral_centroid(
        y, sr, n_fft=n_fft, hop_length=HOP_LENGTH
    )

    # High-frequency energy ratio (>4 kHz)
    high_mask = freqs >= 4000
    high_energy = float(np.sum(magnitude[high_mask, :] ** 2))
    total_energy_spec = float(np.sum(magnitude ** 2))
    high_freq_energy_ratio = high_energy / (total_energy_spec + 1e-10)

    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=n_fft, hop_length=HOP_LENGTH
    )[0]
    spectral_rolloff = float(np.mean(rolloff))

    return {
        "spectral_centroid": spectral_centroid,
        "high_freq_energy_ratio": high_freq_energy_ratio,
        "spectral_rolloff": spectral_rolloff,
    }


# ------------------------------------------------------------------
# 3. Sustain Quality
# ------------------------------------------------------------------

def _extract_sustain_quality(
    y: np.ndarray,
    sr: int,
    n_fft: int,
) -> Dict[str, Any]:
    """
    Sustain quality features measure consistency over the held portion.

    - energy_stability: 1 - (RMS std / RMS mean), excluding onset; higher = more stable
    - pitch_stability: std of detected f0 (lower = more stable); 0 if no pitch found
    """
    # Energy stability — skip first 50 ms (onset)
    onset_samples = int(0.05 * sr)
    sustain = y[onset_samples:]
    if len(sustain) < HOP_LENGTH:
        return {"energy_stability": 0.0, "pitch_stability": 0.0}

    rms_sustain = spec.rms_frames(sustain, hop_length=HOP_LENGTH)
    rms_mean = float(np.mean(rms_sustain))
    rms_std = float(np.std(rms_sustain))
    energy_stability = 1.0 - (rms_std / (rms_mean + 1e-10))
    energy_stability = float(max(0.0, min(1.0, energy_stability)))

    # Pitch stability via pyin
    try:
        f0, voiced_flag, _ = librosa.pyin(
            sustain,
            fmin=float(librosa.note_to_hz("C2")),
            fmax=float(librosa.note_to_hz("C7")),
            sr=sr,
            hop_length=HOP_LENGTH,
        )
        voiced_f0 = f0[~np.isnan(f0)]
        if len(voiced_f0) > 1:
            pitch_stability = float(np.std(voiced_f0))
        else:
            pitch_stability = 0.0
    except Exception:
        pitch_stability = 0.0

    return {
        "energy_stability": energy_stability,
        "pitch_stability": pitch_stability,
    }


# ------------------------------------------------------------------
# 4. Overblowing Detection
# ------------------------------------------------------------------

def _extract_overblowing(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    magnitude: np.ndarray,
) -> Dict[str, Any]:
    """
    Overblowing features detect excess air noise and distortion.

    - spectral_flatness: mean spectral flatness (higher = more noise-like = worse)
    - harmonic_to_noise_ratio: harmonic energy / total energy (higher = cleaner)
    - dynamic_range_compression: 1 - (peak / RMS), compressed signal ≈ overblown
    """
    spectral_flatness = spec.spectral_flatness(y, n_fft=n_fft, hop_length=HOP_LENGTH)

    # HNR on the canonical scale: raw harmonic / total-energy ratio in [0, 1].
    harmonic_to_noise_ratio = spec.harmonic_to_noise_ratio(y)

    # Dynamic range compression indicator
    rms_mean = spec.rms(y, hop_length=HOP_LENGTH)
    rms_peak = float(np.max(np.abs(y)))
    if rms_peak > 1e-10:
        crest_factor = rms_peak / (rms_mean + 1e-10)
        # Normalise: a natural signal has crest factor ~3-10; overblown is <3
        dynamic_range_compression = float(max(0.0, min(1.0, 1.0 - (crest_factor - 1.0) / 9.0)))
    else:
        dynamic_range_compression = 0.0

    return {
        "spectral_flatness": spectral_flatness,
        "harmonic_to_noise_ratio": harmonic_to_noise_ratio,
        "dynamic_range_compression": dynamic_range_compression,
    }


# ------------------------------------------------------------------
# 5. Harmonic Richness
# ------------------------------------------------------------------

def _extract_harmonic_richness(
    y: np.ndarray,
    sr: int,
    magnitude: np.ndarray,
    freqs: np.ndarray,
) -> Dict[str, Any]:
    """
    Harmonic richness features measure the overtone structure.

    - num_harmonics: number of detected harmonics above noise floor
    - harmonic_decay_rate: rate at which harmonic amplitudes decrease (higher = faster decay)
    """
    avg_spectrum = np.mean(magnitude, axis=1)

    # Find fundamental: strongest peak in 50-1000 Hz range
    low_mask = (freqs >= 50) & (freqs <= 1000)
    low_spectrum = avg_spectrum[low_mask]
    low_freqs = freqs[low_mask]

    if len(low_spectrum) == 0 or float(np.max(low_spectrum)) < 1e-10:
        return {"num_harmonics": 0, "harmonic_decay_rate": 0.0}

    fund_idx = int(np.argmax(low_spectrum))
    fundamental = float(low_freqs[fund_idx])
    if fundamental < 50:
        return {"num_harmonics": 0, "harmonic_decay_rate": 0.0}

    # Noise floor: median of the spectrum
    noise_floor = float(np.median(avg_spectrum)) * 2.0

    # Count harmonics above noise floor
    harmonic_amps = []
    for n in range(1, 21):
        harmonic_freq = fundamental * n
        if harmonic_freq >= freqs[-1]:
            break
        freq_idx = int(np.argmin(np.abs(freqs - harmonic_freq)))
        amp = float(avg_spectrum[freq_idx])
        if amp > noise_floor:
            harmonic_amps.append(amp)
        else:
            break  # stop at first missing harmonic

    num_harmonics = len(harmonic_amps)

    # Harmonic decay rate: fit log-linear decay to harmonic amplitudes
    harmonic_decay_rate = 0.0
    if num_harmonics >= 3:
        log_amps = np.log(np.array(harmonic_amps) + 1e-10)
        indices = np.arange(1, num_harmonics + 1, dtype=float)
        # Linear fit: log(amp) = a - decay_rate * n
        coeffs = np.polyfit(indices, log_amps, 1)
        harmonic_decay_rate = float(-coeffs[0])  # positive = decaying

    return {
        "num_harmonics": num_harmonics,
        "harmonic_decay_rate": harmonic_decay_rate,
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_audio(
    audio: Union[str, np.ndarray], sr: int
) -> tuple:
    """Load audio from file path or pass through ndarray."""
    if isinstance(audio, str):
        y, sr_loaded = librosa.load(audio, sr=sr, mono=True)
        return y, sr_loaded
    elif isinstance(audio, np.ndarray):
        return audio, sr
    else:
        raise TypeError(f"Expected str path or np.ndarray, got {type(audio)}")


def _empty_features() -> Dict[str, Any]:
    """Return a zero-valued feature dict when audio is empty."""
    return {
        "attack_time": 0.0,
        "transient_energy_ratio": 0.0,
        "onset_spectral_spread": 0.0,
        "spectral_centroid": 0.0,
        "high_freq_energy_ratio": 0.0,
        "spectral_rolloff": 0.0,
        "energy_stability": 0.0,
        "pitch_stability": 0.0,
        "spectral_flatness": 0.0,
        "harmonic_to_noise_ratio": 0.0,
        "dynamic_range_compression": 0.0,
        "num_harmonics": 0,
        "harmonic_decay_rate": 0.0,
    }
