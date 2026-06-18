"""Audio analysis service — wraps the generic AudioAnalyzer from services/rag/.

Portfolio snapshot: this exposes only the generic feature-extraction analyzer
used by the chat audio path. The specialized scoring facades (articulation,
played-passage, register) and their rubrics are excluded from this snapshot.
"""

import logging
import sys
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Add services/rag to the path so we can import audio_analyzer
_rag_dir = os.path.join(os.path.dirname(__file__), "..", "..", "services", "rag")
if os.path.isdir(_rag_dir) and _rag_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_rag_dir))


def analyze_audio(file_path: str) -> Dict[str, Any]:
    """
    Analyze an audio file and return structured results.

    Args:
        file_path: Absolute path to the audio file on disk.

    Returns:
        Dict with keys: pitch_data, tone_quality, compression_metrics,
        buzz_quality, spectral_data, duration, tempo.

    Raises:
        ValueError: For silence, non-trumpet audio, or clips under 0.5 s.
        ImportError: If librosa is not installed.
    """
    from audio_analyzer import AudioAnalyzer, LIBROSA_AVAILABLE
    import librosa as _librosa

    if not LIBROSA_AVAILABLE:
        raise ImportError(
            "librosa is required for audio analysis. "
            "Install with: pip install librosa soundfile"
        )

    # Quick duration check before heavy analysis
    y, sr = _librosa.load(file_path, sr=22050)
    duration = _librosa.get_duration(y=y, sr=sr)

    if duration < 0.5:
        raise ValueError(
            f"Audio clip is too short ({duration:.2f}s). "
            "Please provide a recording of at least 0.5 seconds."
        )

    # Check for near-silence
    import numpy as _np
    rms = _np.sqrt(_np.mean(y ** 2))
    if rms < 1e-4:
        raise ValueError(
            "The recording appears to be silence. "
            "Please upload a recording with audible trumpet playing."
        )

    # Run the full analysis
    analyzer = AudioAnalyzer(sample_rate=22050)
    raw = analyzer.analyze_audio(file_path, extract_notes=True, extract_characteristics=True)

    notes = raw.get("notes", [])
    chars = raw.get("characteristics", {})

    pitch_data = {
        "notes": [
            {
                "note_name": n.get("note_name", ""),
                "pitch_hz": n.get("pitch_hz", 0.0),
                "start_time": round(n.get("start_time", 0.0), 3),
                "end_time": round(n.get("end_time", 0.0), 3),
                "duration": round(n.get("duration", 0.0), 3),
                "confidence": round(n.get("confidence", 0.0), 3),
            }
            for n in notes
        ],
        "note_count": raw.get("note_count", len(notes)),
        "tempo_bpm": raw.get("tempo"),
    }

    tone_quality = {
        "spectral_centroid": chars.get("spectral_centroid", {}),
        "harmonic_to_noise_ratio": chars.get("harmonic_to_noise_ratio", {}),
        "mfcc": chars.get("mfcc", {}),
        "sustain_quality": chars.get("sustain_quality", {}),
    }

    compression_metrics = {
        "rms_energy": chars.get("rms_energy", {}),
        "attack_time": chars.get("attack_time", {}),
    }

    buzz_quality = {
        "sawtooth_similarity": chars.get("sawtooth_similarity", {}),
        "spectral_flatness": chars.get("spectral_flatness", {}),
        "high_frequency_energy_ratio": chars.get("high_frequency_energy_ratio", {}),
    }

    spectral_data = {
        "spectral_rolloff": chars.get("spectral_rolloff", {}),
        "spectral_bandwidth": chars.get("spectral_bandwidth", {}),
        "zero_crossing_rate": chars.get("zero_crossing_rate", {}),
    }

    return {
        "pitch_data": pitch_data,
        "tone_quality": tone_quality,
        "compression_metrics": compression_metrics,
        "buzz_quality": buzz_quality,
        "spectral_data": spectral_data,
        "duration": raw.get("duration", duration),
        "tempo": raw.get("tempo"),
    }
