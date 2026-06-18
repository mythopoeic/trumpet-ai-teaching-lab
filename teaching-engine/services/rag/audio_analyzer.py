"""
Audio analysis service for extracting pitch, duration, and characteristics from audio files.
Designed for matching audio recordings to sheet music.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import json

logger = logging.getLogger(__name__)

try:
    import librosa
    import numpy as np
    from audio import spectral_features as spec
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not available. Install with: pip install librosa soundfile")


class AudioAnalyzer:
    """
    Analyzes audio files to extract musical characteristics:
    - Pitch/notes (fundamental frequency detection)
    - Note durations
    - Timbre characteristics
    - Dynamics
    - Timing
    """
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        """
        Initialize audio analyzer.
        
        Args:
            sample_rate: Target sample rate for analysis (default 22050 Hz)
            hop_length: Number of samples between frames (default 512)
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required. Install with: pip install librosa soundfile")
        
        self.sample_rate = sample_rate
        self.hop_length = hop_length
    
    def analyze_audio(
        self,
        audio_path: str,
        extract_notes: bool = True,
        extract_characteristics: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze audio file and extract musical information.
        
        Args:
            audio_path: Path to audio file (MP3, WAV, etc.)
            extract_notes: Whether to extract pitch/note information
            extract_characteristics: Whether to extract audio characteristics
            
        Returns:
            Dictionary with analysis results:
            {
                'notes': List of detected notes with timing,
                'characteristics': Audio characteristics,
                'duration': Total audio duration,
                'tempo': Estimated tempo,
                'metadata': File metadata
            }
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Analyzing audio: {Path(audio_path).name}")
        
        # Load audio
        logger.info("  Loading audio file...")
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = librosa.get_duration(y=y, sr=sr)
        logger.info(f"  ✓ Loaded: {duration:.2f}s at {sr}Hz")
        
        results = {
            'filename': Path(audio_path).name,
            'duration': duration,
            'sample_rate': sr,
            'metadata': {}
        }
        
        # Extract tempo
        logger.info("  Extracting tempo...")
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
            # Convert tempo to float if it's a numpy array/scalar
            if hasattr(tempo, 'item'):
                tempo = tempo.item()
            tempo_float = float(tempo)
            results['tempo'] = tempo_float
            results['beat_frames'] = beats.tolist() if hasattr(beats, 'tolist') else list(beats)
            logger.info(f"  ✓ Tempo: {tempo_float:.1f} BPM")
        except Exception as e:
            logger.warning(f"  ⚠ Could not estimate tempo: {e}")
            results['tempo'] = None
        
        # Extract notes/pitch
        if extract_notes:
            logger.info("  Extracting notes/pitch (this may take a moment)...")
            notes = self._extract_notes(y, sr)
            results['notes'] = notes
            results['note_count'] = len(notes)
            logger.info(f"  ✓ Extracted {len(notes)} notes")
        else:
            logger.info("  Skipping note extraction (extract_notes=False)")
        
        # Extract characteristics
        if extract_characteristics:
            logger.info("  Extracting audio characteristics...")
            characteristics = self._extract_characteristics(y, sr)
            results['characteristics'] = characteristics
            logger.info("  ✓ Characteristics extracted")
        else:
            logger.info("  Skipping characteristics extraction")
        
        return results
    
    def _extract_notes(self, y: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """
        Extract notes (pitches) and their durations from audio.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            List of note dictionaries:
            {
                'start_time': float (seconds),
                'end_time': float (seconds),
                'duration': float (seconds),
                'pitch_hz': float (frequency in Hz),
                'note_name': str (e.g., 'C4', 'Bb3'),
                'confidence': float (0-1)
            }
        """
        notes = []
        
        # Extract fundamental frequency using librosa
        # Use pyin (probabilistic YIN) for better pitch tracking
        # For pedal tones, we need to go even lower - pedal tones can be Bb1, A1, G1, etc.
        try:
            # Use a longer frame length for better low-frequency resolution (pedal tones)
            # frame_length should be a power of 2 and at least 2x hop_length
            # Longer frames = better frequency resolution for low frequencies
            # For pedal tones (very low), we need even longer frames
            frame_length = max(8192, self.hop_length * 8)  # Use 8192 for better low-frequency resolution
            # Ensure it's a power of 2 (required by librosa)
            frame_length = int(2 ** np.ceil(np.log2(frame_length)))
            
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=librosa.note_to_hz('Ab0'),  # Even lower - Ab0 to catch all pedal tones including F1, E1
                fmax=librosa.note_to_hz('C7'),  # Up to C7 (high trumpet range)
                sr=sr,
                hop_length=self.hop_length,
                frame_length=frame_length  # Longer frame for better low-frequency resolution
                # Note: threshold parameter not available in all librosa versions
                # Lower fmin and longer frame_length help detect quiet pedal tones
            )
            
            # Calculate RMS energy to detect silence/breaks
            # Use same frame_length as pyin for consistency
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length, frame_length=frame_length)[0]
            # Use a more conservative threshold - bottom 30% is silence (more lenient for pedal tones)
            # Pedal tones are often quieter, so we need a lower threshold
            rms_percentile = np.percentile(rms[rms > 0], 30)
            rms_min = np.max(rms) * 0.06  # 6% of max energy as absolute minimum (lower for very quiet pedal notes)
            rms_threshold = max(rms_percentile, rms_min)
            
            times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=self.hop_length)
            
            # Parameters for note segmentation
            # For Roll-Out exercises with double pedals, notes are long and sustained
            min_note_duration = 0.2   # Minimum note duration in seconds (reduced slightly to catch pedal notes)
            max_gap_duration = 0.8    # Maximum gap between notes before splitting (longer for pedal exercises)
            pitch_change_threshold_semitones = 3.5  # Minimum pitch change in semitones (increased to reduce false splits)
            min_confidence = 0.2      # Lower confidence threshold to catch pedal tones which may have lower confidence
            min_confidence_pedal = 0.1  # Even lower threshold for very low frequencies (pedal tones)
            pedal_freq_threshold = librosa.note_to_hz('C2')  # Below C2 is considered pedal range
            
            # Segment into individual notes with silence/gap detection
            current_note = None
            silence_start = None
            
            for i, (freq, is_voiced, prob) in enumerate(zip(f0, voiced_flag, voiced_probs)):
                time = times[i]
                frame_energy = rms[min(i, len(rms) - 1)]
                is_silent = frame_energy < rms_threshold
                
                # Filter by confidence and silence
                # Use lower confidence threshold for pedal tones (very low frequencies)
                is_pedal_range = freq < pedal_freq_threshold
                confidence_threshold = min_confidence_pedal if is_pedal_range else min_confidence
                confidence_ok = (not np.isnan(prob)) and (prob >= confidence_threshold)
                
                # For pedal tones, also relax the silence check (they're often quieter)
                is_silent_check = is_silent
                if is_pedal_range:
                    # For pedal tones, only consider it silent if energy is very low
                    pedal_rms_threshold = rms_threshold * 0.7  # 30% lower threshold for pedal range
                    is_silent_check = frame_energy < pedal_rms_threshold
                
                if is_voiced and not np.isnan(freq) and not is_silent_check and confidence_ok:
                    note_name = librosa.hz_to_note(freq, unicode=False)
                    
                    # Check if we should start a new note
                    should_start_new = False
                    
                    if current_note is None:
                        # No current note - start one
                        should_start_new = True
                    else:
                        # Check pitch change (use semitones for better musical accuracy)
                        # Convert Hz difference to approximate semitones
                        avg_freq = (freq + current_note['pitch_hz']) / 2
                        if avg_freq > 0 and freq > 0:
                            semitones_diff = abs(12 * np.log2(freq / current_note['pitch_hz']))
                            
                            # Special handling for transitions to pedal tones (very low frequencies)
                            # If we're going from a high note (octave 3+) to a very low note (octave 1),
                            # this is likely a transition to a pedal tone, not a continuation
                            current_is_high = current_note['pitch_hz'] >= pedal_freq_threshold
                            new_is_pedal = freq < pedal_freq_threshold
                            
                            # If transitioning from high to pedal, always create new note (huge pitch jump is expected)
                            if current_is_high and new_is_pedal and semitones_diff > 12:  # More than an octave
                                should_start_new = True
                            # Otherwise use normal threshold
                            elif semitones_diff > pitch_change_threshold_semitones:
                                should_start_new = True
                    
                    if should_start_new:
                        # End previous note if it meets minimum duration
                        if current_note:
                            current_note['end_time'] = time
                            current_note['duration'] = current_note['end_time'] - current_note['start_time']
                            if current_note['duration'] >= min_note_duration:
                                notes.append(current_note)
                            # else: discard note as too short (likely artifact)
                        
                        # Start new note
                        current_note = {
                            'start_time': time,
                            'pitch_hz': float(freq),
                            'note_name': note_name,
                            'confidence': float(prob) if not np.isnan(prob) else 0.5,
                            'pitch_samples': [float(freq)]  # Track multiple pitch samples for averaging
                        }
                        silence_start = None
                    else:
                        # Continue current note - update average pitch
                        current_note['pitch_samples'].append(float(freq))
                        # For pedal tones, use mean instead of median (more responsive to actual pitch)
                        if is_pedal_range:
                            current_note['pitch_hz'] = float(np.mean(current_note['pitch_samples']))
                        else:
                            current_note['pitch_hz'] = float(np.median(current_note['pitch_samples']))
                        current_note['note_name'] = librosa.hz_to_note(current_note['pitch_hz'], unicode=False)
                        current_note['confidence'] = max(current_note['confidence'], float(prob) if not np.isnan(prob) else 0.5)
                        silence_start = None
                else:
                    # Silence or no pitch - track gap duration
                    # But be more lenient for pedal tones (they might have brief gaps or low confidence)
                    if current_note:
                        current_is_pedal = current_note.get('pitch_hz', 0) < pedal_freq_threshold
                        
                        if silence_start is None:
                            silence_start = time
                        else:
                            gap_duration = time - silence_start
                            # Pedal tones can have longer gaps - they're harder to track consistently
                            # Also allow brief gaps in regular notes (might be a breath or transition)
                            max_gap = max_gap_duration * 1.5 if current_is_pedal else max_gap_duration * 1.2
                            
                            # If gap is too long, end the note
                            if gap_duration > max_gap:
                                current_note['end_time'] = silence_start  # End note at start of silence
                                current_note['duration'] = current_note['end_time'] - current_note['start_time']
                                
                                # Lower minimum duration for pedal tones (they might be shorter due to detection issues)
                                min_dur = min_note_duration * 0.7 if current_is_pedal else min_note_duration
                                if current_note['duration'] >= min_dur:
                                    # Clean up pitch_samples before storing
                                    if 'pitch_samples' in current_note:
                                        del current_note['pitch_samples']
                                    notes.append(current_note)
                                current_note = None
                                silence_start = None
                            # If gap is shorter, continue tracking (might be a brief silence in a sustained note)
            
            # Finalize last note if exists
            if current_note:
                # End at last valid time (before silence if there was a gap)
                end_time = silence_start if silence_start else (times[-1] if len(times) > 0 else current_note['start_time'] + 0.5)
                current_note['end_time'] = end_time
                current_note['duration'] = current_note['end_time'] - current_note['start_time']
                # Clean up pitch_samples before storing
                if 'pitch_samples' in current_note:
                    del current_note['pitch_samples']
                if current_note['duration'] >= min_note_duration:
                    notes.append(current_note)
            
            # Post-process: merge notes that are very close together and similar in pitch
            # (handles cases where a brief gap was incorrectly split)
            notes = self._merge_close_notes(notes, max_gap=0.5, max_pitch_diff_semitones=1.5)
            
            # Additional post-processing: remove very short notes that are likely artifacts
            # But be more lenient with pedal tones (they might be shorter due to detection issues)
            filtered_notes = []
            for note in notes:
                freq = note.get('pitch_hz', 0)
                duration = note.get('duration', 0)
                is_pedal = freq < pedal_freq_threshold
                min_dur = min_note_duration * 0.7 if is_pedal else min_note_duration  # 30% shorter minimum for pedals
                if duration >= min_dur:
                    filtered_notes.append(note)
                else:
                    logger.debug(f"  Filtered out short note: {note.get('note_name', '?')} ({duration:.2f}s, pedal: {is_pedal})")
            notes = filtered_notes
            
            # Sort notes by start time (in case merging changed order)
            notes.sort(key=lambda n: n.get('start_time', 0))
            
            # Post-processing: Search for missing pedal tones using spectral analysis
            # pyin might miss very quiet/low pedal tones, so we'll look for them explicitly
            logger.info(f"  Found {len(notes)} notes from pyin, searching for missing pedal tones...")
            notes = self._detect_missing_pedal_tones(y, sr, notes, times, pedal_freq_threshold, rms, rms_threshold)
            
            # Re-sort after adding pedal tones
            notes.sort(key=lambda n: n.get('start_time', 0))
            
            # Log detected note ranges for debugging
            if notes:
                note_names = [n.get('note_name', '?') for n in notes]
                pedal_notes = [n for n in note_names if '1' in n or '0' in n]
                logger.info(f"  Extracted {len(notes)} notes total after pedal tone detection (min duration: {min_note_duration}s)")
                if pedal_notes:
                    logger.info(f"    Pedal tones detected: {len(pedal_notes)} ({', '.join(set(pedal_notes)[:5])}{'...' if len(set(pedal_notes)) > 5 else ''})")
                else:
                    logger.warning(f"    ⚠ No pedal tones (octave 1) detected - may need further tuning")
        
        except Exception as e:
            logger.warning(f"  Warning: Error extracting notes with pyin: {e}")
            logger.info("  Falling back to simple pitch detection...")
            # Fallback to simpler pitch detection
            notes = self._extract_notes_simple(y, sr)
        
        return notes
    
    def _detect_missing_pedal_tones(
        self,
        y: np.ndarray,
        sr: int,
        existing_notes: List[Dict[str, Any]],
        times: np.ndarray,
        pedal_freq_threshold: float,
        rms: np.ndarray,
        rms_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Post-process to detect missing pedal tones using spectral analysis.
        Pedal tones are very low frequency (44-93 Hz) and may be missed by pyin.
        
        Args:
            y: Audio signal
            sr: Sample rate
            existing_notes: Notes already detected by pyin
            times: Time array for frames
            pedal_freq_threshold: Frequency below which is considered pedal range
            rms: RMS energy array
            rms_threshold: RMS threshold for silence detection
            
        Returns:
            List of notes including newly detected pedal tones
        """
        if len(existing_notes) == 0:
            return existing_notes
        
        new_pedal_notes = []
        min_note_duration = 0.2  # Minimum duration for a valid note
        
        # Look for gaps where pedal tones might be missing
        # Specifically, look after high notes (which might be followed by pedal tones)
        for i in range(len(existing_notes) - 1):
            current_note = existing_notes[i]
            next_note = existing_notes[i + 1]
            
            # Check if current note is high (octave 3+) and there's a gap before next note
            if current_note['pitch_hz'] >= pedal_freq_threshold:
                gap_start = current_note['end_time']
                gap_end = next_note['start_time']
                gap_duration = gap_end - gap_start
                
                # Look for gaps that are at least 0.5s and up to 5s (whole note duration)
                # This is where a pedal tone might be
                if 0.5 <= gap_duration <= 5.0:
                    # Extract audio segment for this gap
                    start_sample = int(gap_start * sr)
                    end_sample = int(gap_end * sr)
                    
                    if start_sample < len(y) and end_sample <= len(y) and end_sample > start_sample:
                        segment = y[start_sample:end_sample]
                        
                        # Check if there's significant energy (not silence)
                        segment_rms = np.sqrt(np.mean(segment**2))
                        if segment_rms > rms_threshold * 0.5:  # At least 50% of silence threshold
                            # Use autocorrelation to find fundamental frequency
                            f0_pedal = self._find_fundamental_autocorr(segment, sr, fmin=librosa.note_to_hz('F0'), fmax=pedal_freq_threshold)
                            
                            if f0_pedal is not None and f0_pedal < pedal_freq_threshold:
                                # Found a potential pedal tone!
                                note_name = librosa.hz_to_note(f0_pedal, unicode=False)
                                
                                # Check if this note doesn't already exist (avoid duplicates)
                                is_duplicate = False
                                for existing in existing_notes + new_pedal_notes:
                                    existing_start = existing.get('start_time', 0)
                                    existing_end = existing.get('end_time', 0)
                                    # Check if times overlap significantly
                                    if not (gap_end < existing_start or gap_start > existing_end):
                                        existing_freq = existing.get('pitch_hz', 0)
                                        if existing_freq > 0:
                                            freq_diff_semitones = abs(12 * np.log2(f0_pedal / existing_freq))
                                            if freq_diff_semitones < 2.0:  # Within 2 semitones
                                                is_duplicate = True
                                                break
                                
                                if not is_duplicate:
                                    pedal_note = {
                                        'start_time': gap_start,
                                        'end_time': gap_end,
                                        'duration': gap_duration,
                                        'pitch_hz': float(f0_pedal),
                                        'note_name': note_name,
                                        'confidence': 0.6,  # Lower confidence for post-processed detection
                                        'detection_method': 'spectral_analysis'
                                    }
                                    new_pedal_notes.append(pedal_note)
                                    logger.info(f"    Found missing pedal tone: {note_name} ({f0_pedal:.1f} Hz) at {gap_start:.2f}-{gap_end:.2f}s")
        
        # Also check the end of the recording (last note might be followed by a pedal tone)
        if len(existing_notes) > 0:
            last_note = existing_notes[-1]
            if last_note['pitch_hz'] >= pedal_freq_threshold:  # Last note is high
                gap_start = last_note['end_time']
                gap_end = len(y) / sr  # End of recording
                gap_duration = gap_end - gap_start
                
                if gap_duration >= min_note_duration and gap_duration <= 5.0:
                    start_sample = int(gap_start * sr)
                    end_sample = min(int(gap_end * sr), len(y))
                    
                    if end_sample > start_sample:
                        segment = y[start_sample:end_sample]
                        segment_rms = np.sqrt(np.mean(segment**2))
                        
                        if segment_rms > rms_threshold * 0.5:
                            f0_pedal = self._find_fundamental_autocorr(segment, sr, fmin=librosa.note_to_hz('F0'), fmax=pedal_freq_threshold)
                            
                            if f0_pedal is not None and f0_pedal < pedal_freq_threshold:
                                note_name = librosa.hz_to_note(f0_pedal, unicode=False)
                                pedal_note = {
                                    'start_time': gap_start,
                                    'end_time': gap_end,
                                    'duration': gap_duration,
                                    'pitch_hz': float(f0_pedal),
                                    'note_name': note_name,
                                    'confidence': 0.6,
                                    'detection_method': 'spectral_analysis'
                                }
                                new_pedal_notes.append(pedal_note)
                                logger.info(f"    Found missing pedal tone at end: {note_name} ({f0_pedal:.1f} Hz) at {gap_start:.2f}-{gap_end:.2f}s")
        
        # Combine existing notes with newly found pedal tones
        all_notes = existing_notes + new_pedal_notes
        logger.info(f"  Added {len(new_pedal_notes)} pedal tones via spectral analysis")
        
        return all_notes
    
    def _find_fundamental_autocorr(
        self,
        segment: np.ndarray,
        sr: int,
        fmin: float = 44.0,
        fmax: float = 330.0
    ) -> Optional[float]:
        """
        Find fundamental frequency using autocorrelation.
        This is more reliable for very low frequencies than pyin.
        
        Args:
            segment: Audio segment
            sr: Sample rate
            fmin: Minimum frequency to search (Hz)
            fmax: Maximum frequency to search (Hz)
            
        Returns:
            Fundamental frequency in Hz, or None if not found
        """
        if len(segment) < sr / fmin:  # Need at least one period
            return None
        
        # Normalize the segment
        segment = segment - np.mean(segment)
        if np.max(np.abs(segment)) > 0:
            segment = segment / np.max(np.abs(segment))
        else:
            return None
        
        # Compute autocorrelation
        autocorr = np.correlate(segment, segment, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Take only positive lags
        
        # Find the period corresponding to fmin and fmax
        min_period = int(sr / fmax)
        max_period = min(int(sr / fmin), len(autocorr) // 2)
        
        if max_period <= min_period:
            return None
        
        # Look for peaks in the autocorrelation in the valid range
        valid_range = autocorr[min_period:max_period]
        
        if len(valid_range) == 0:
            return None
        
        # Find the peak (strongest correlation)
        peak_idx = np.argmax(valid_range) + min_period
        peak_value = autocorr[peak_idx]
        
        # Check if peak is significant (at least 20% of max autocorr)
        max_autocorr = np.max(autocorr)
        if peak_value < max_autocorr * 0.2:
            return None
        
        # Convert lag to frequency
        if peak_idx > 0:
            fundamental_freq = sr / peak_idx
            # Ensure it's in valid range
            if fmin <= fundamental_freq <= fmax:
                return fundamental_freq
        
        return None
    
    def _merge_close_notes(self, notes: List[Dict[str, Any]], max_gap: float = 0.5, max_pitch_diff_semitones: float = 1.5) -> List[Dict[str, Any]]:
        """
        Merge notes that are very close together in time and similar in pitch.
        This handles cases where brief breaks or artifacts split a single note.
        
        Args:
            notes: List of note dictionaries
            max_gap: Maximum gap in seconds to consider merging (default 0.5s, increased for pedal exercises)
            max_pitch_diff_semitones: Maximum pitch difference in semitones to merge (default 1.5, allows slight pitch variations)
            
        Returns:
            Merged list of notes
        """
        if len(notes) <= 1:
            return notes
        
        merged = []
        current = notes[0].copy()
        
        for next_note in notes[1:]:
            gap = next_note['start_time'] - current['end_time']
            
            # Check if notes should be merged
            should_merge = False
            
            if gap <= max_gap:
                # Calculate pitch difference in semitones
                if current['pitch_hz'] > 0 and next_note['pitch_hz'] > 0:
                    semitones_diff = abs(12 * np.log2(next_note['pitch_hz'] / current['pitch_hz']))
                    
                    if semitones_diff <= max_pitch_diff_semitones:
                        should_merge = True
                    # Also merge if gap is very small (< 0.2s) even if pitch differs slightly more
                    # (might be a brief artifact or transition during a sustained note)
                    elif gap < 0.2 and semitones_diff <= (max_pitch_diff_semitones + 1.0):
                        should_merge = True
            
            if should_merge:
                # Merge: extend current note to include next
                current['end_time'] = next_note['end_time']
                current['duration'] = current['end_time'] - current['start_time']
                # Update pitch to weighted average (by duration)
                current_dur = current.get('duration', 0) or 0.1
                next_dur = next_note.get('duration', 0) or 0.1
                total_dur = current_dur + next_dur
                if total_dur > 0:
                    # Weight by duration for more accurate pitch
                    current['pitch_hz'] = (current['pitch_hz'] * current_dur + next_note['pitch_hz'] * next_dur) / total_dur
                    current['note_name'] = librosa.hz_to_note(current['pitch_hz'], unicode=False)
                    current['confidence'] = max(current.get('confidence', 0.5), next_note.get('confidence', 0.5))
                continue
            
            # Can't merge - add current and move to next
            merged.append(current)
            current = next_note.copy()
        
        # Add last note
        merged.append(current)
        
        return merged
    
    def _extract_notes_simple(self, y: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Fallback simple pitch detection."""
        notes = []
        
        # Use autocorrelation for fundamental frequency
        frame_length = 2048
        hop_length = self.hop_length
        
        times = librosa.frames_to_time(
            np.arange(0, len(y) - frame_length, hop_length),
            sr=sr,
            hop_length=hop_length
        )
        
        for i, start in enumerate(range(0, len(y) - frame_length, hop_length)):
            frame = y[start:start + frame_length]
            
            # Autocorrelation
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peak (excluding DC component)
            min_period = int(sr / librosa.note_to_hz('C7'))  # Highest note
            max_period = int(sr / librosa.note_to_hz('C2'))  # Lowest note
            
            if max_period < len(autocorr):
                autocorr_region = autocorr[min_period:max_period]
                if len(autocorr_region) > 0:
                    peak_idx = np.argmax(autocorr_region) + min_period
                    freq = sr / peak_idx if peak_idx > 0 else 0
                    
                    if freq > 0:
                        note_name = librosa.hz_to_note(freq, unicode=False)
                        
                        # Group into notes (simplified)
                        if not notes or abs(freq - notes[-1]['pitch_hz']) > 50:
                            notes.append({
                                'start_time': times[i] if i < len(times) else i * hop_length / sr,
                                'end_time': None,  # Will be set by next note
                                'duration': None,
                                'pitch_hz': float(freq),
                                'note_name': note_name,
                                'confidence': 0.6
                            })
        
        # Set end times
        for i in range(len(notes) - 1):
            notes[i]['end_time'] = notes[i+1]['start_time']
            notes[i]['duration'] = notes[i]['end_time'] - notes[i]['start_time']
        
        if notes:
            notes[-1]['end_time'] = times[-1] if len(times) > 0 else len(y) / sr
            notes[-1]['duration'] = notes[-1]['end_time'] - notes[-1]['start_time']
        
        return notes
    
    def _extract_characteristics(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract audio characteristics that indicate quality.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary with characteristics:
            {
                'rms_energy': Root mean square energy (dynamics),
                'spectral_centroid': Brightness/timbre,
                'spectral_rolloff': High frequency content,
                'zero_crossing_rate': Articulation,
                'mfcc': Mel-frequency cepstral coefficients (timbre),
                'attack_time': Note attack characteristics,
                'sustain_quality': Consistency during sustain
            }
        """
        characteristics = {}
        
        # RMS Energy (dynamics)
        rms = spec.rms_frames(y, hop_length=self.hop_length)
        characteristics['rms_energy'] = {
            'mean': float(np.mean(rms)),
            'std': float(np.std(rms)),
            'max': float(np.max(rms)),
            'min': float(np.min(rms))
        }
        
        # Spectral features
        spectral_centroids = spec.spectral_centroid_frames(y, sr, hop_length=self.hop_length)
        characteristics['spectral_centroid'] = {
            'mean': float(np.mean(spectral_centroids)),
            'std': float(np.std(spectral_centroids))
        }
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)[0]
        characteristics['spectral_rolloff'] = {
            'mean': float(np.mean(spectral_rolloff)),
            'std': float(np.std(spectral_rolloff))
        }
        
        # Zero crossing rate (articulation)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
        characteristics['zero_crossing_rate'] = {
            'mean': float(np.mean(zcr)),
            'std': float(np.std(zcr))
        }
        
        # MFCC (timbre)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length)
        characteristics['mfcc'] = {
            'mean': [float(x) for x in np.mean(mfccs, axis=1)],
            'std': [float(x) for x in np.std(mfccs, axis=1)]
        }
        
        # Harmonic analysis for metallic/buzzy tone quality
        # Extract harmonic and percussive components
        try:
            y_harmonic, _ = librosa.effects.hpss(y)

            # Compute spectral features for harmonic component
            stft_harmonic = librosa.stft(y_harmonic, hop_length=self.hop_length)
            magnitude_harmonic = np.abs(stft_harmonic)

            # Harmonic-to-noise ratio on the canonical scale: raw harmonic /
            # total-energy ratio in [0, 1] (higher = more tonal, less noisy).
            # NOTE: deliberate scale change — this used to be a log10 value
            # under the same key, which a downstream consumer silently mixed
            # with the spit-buzz extractor's raw ratio. Both keys now hold the
            # one canonical raw ratio; the dict shape is preserved.
            hnr = spec.harmonic_to_noise_ratio(y)
            characteristics['harmonic_to_noise_ratio'] = {
                'mean': float(hnr),
                'raw': float(hnr)
            }
            
            # Spectral bandwidth (wider = more buzzy/metallic)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=self.hop_length)[0]
            characteristics['spectral_bandwidth'] = {
                'mean': float(np.mean(spectral_bandwidth)),
                'std': float(np.std(spectral_bandwidth))
            }
            
            # Spectral flatness (lower = more tonal, higher = more noise-like)
            # But for sawtooth-like sounds, we want moderate flatness with strong harmonics
            spectral_flatness = spec.spectral_flatness_frames(y, hop_length=self.hop_length)
            characteristics['spectral_flatness'] = {
                'mean': float(np.mean(spectral_flatness)),
                'std': float(np.std(spectral_flatness))
            }
            
            # High-frequency energy (above 2kHz) - metallic sounds have more high-frequency content
            stft_full = librosa.stft(y, hop_length=self.hop_length)
            freqs = librosa.fft_frequencies(sr=sr)
            magnitude_full = np.abs(stft_full)
            
            # Find indices for frequencies above 2kHz
            high_freq_mask = freqs >= 2000
            high_freq_energy = np.sum(magnitude_full[high_freq_mask, :] ** 2, axis=0)
            total_energy = np.sum(magnitude_full ** 2, axis=0)
            high_freq_ratio = high_freq_energy / (total_energy + 1e-10)
            
            characteristics['high_frequency_energy_ratio'] = {
                'mean': float(np.mean(high_freq_ratio)),
                'std': float(np.std(high_freq_ratio))
            }
            
            # Sawtooth approximation: Check how well harmonics follow 1/n pattern
            # For a sawtooth wave, harmonic amplitudes decrease as 1/n
            # We'll analyze the first 10 harmonics
            sawtooth_score = self._calculate_sawtooth_similarity(magnitude_harmonic, sr, freqs)
            characteristics['sawtooth_similarity'] = {
                'score': float(sawtooth_score),
                'description': 'How well the spectrum matches a sawtooth wave pattern (higher = more metallic/buzzy)'
            }
            
        except Exception as e:
            logger.warning(f"  ⚠ Could not compute harmonic analysis: {e}")
            characteristics['harmonic_to_noise_ratio'] = {'mean': 0.0, 'raw': 0.0}
            characteristics['spectral_bandwidth'] = {'mean': 0.0, 'std': 0.0}
            characteristics['spectral_flatness'] = {'mean': 0.0, 'std': 0.0}
            characteristics['high_frequency_energy_ratio'] = {'mean': 0.0, 'std': 0.0}
            characteristics['sawtooth_similarity'] = {'score': 0.0, 'description': 'Analysis failed'}
        
        # Attack time (onset detection)
        try:
            onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=self.hop_length, units='time')
            if len(onsets) > 0:
                # Measure attack time (time to reach 90% of peak after onset)
                attack_times = []
                for onset in onsets[:min(10, len(onsets))]:  # Sample first 10 onsets
                    onset_frame = librosa.time_to_frames(onset, sr=sr, hop_length=self.hop_length)
                    if onset_frame < len(rms):
                        peak_idx = np.argmax(rms[onset_frame:onset_frame+20]) + onset_frame
                        if peak_idx < len(rms):
                            peak_val = rms[peak_idx]
                            threshold = peak_val * 0.9
                            attack_frames = np.where(rms[onset_frame:peak_idx] >= threshold)[0]
                            if len(attack_frames) > 0:
                                attack_time = librosa.frames_to_time(attack_frames[0], sr=sr, hop_length=self.hop_length)
                                attack_times.append(attack_time)
                
                if attack_times:
                    characteristics['attack_time'] = {
                        'mean': float(np.mean(attack_times)),
                        'std': float(np.std(attack_times))
                    }
        except Exception as e:
            print(f"  Warning: Could not compute attack time: {e}")
        
        # Sustain quality (consistency during held notes)
        # Look for periods of stable pitch and energy
        try:
            # Use a sliding window to find stable regions
            window_size = int(sr * 0.5)  # 0.5 second windows
            stability_scores = []
            
            for i in range(0, len(y) - window_size, window_size // 2):
                window = y[i:i+window_size]
                window_rms = np.mean(librosa.feature.rms(y=window)[0])
                window_zcr = np.mean(librosa.feature.zero_crossing_rate(window)[0])
                
                # Stability = low variance in RMS and ZCR
                stability = 1.0 / (1.0 + window_rms * 10 + window_zcr * 100)
                stability_scores.append(float(stability))
            
            if stability_scores:
                characteristics['sustain_quality'] = {
                    'mean': float(np.mean(stability_scores)),
                    'std': float(np.std(stability_scores))
                }
        except Exception as e:
            print(f"  Warning: Could not compute sustain quality: {e}")
        
        return characteristics
    
    def _calculate_sawtooth_similarity(self, magnitude_spectrum: np.ndarray, sr: int, freqs: np.ndarray) -> float:
        """
        Calculate how well the audio spectrum matches a sawtooth wave pattern.
        
        A sawtooth wave has harmonics that decrease in amplitude as 1/n (where n is harmonic number).
        This creates a buzzy, metallic sound - ideal for Callet's double pedals.
        
        Args:
            magnitude_spectrum: Magnitude spectrogram (freq_bins x time_frames)
            sr: Sample rate
            freqs: Frequency array corresponding to spectrum rows
            
        Returns:
            Score 0-1 indicating sawtooth similarity (higher = more metallic/buzzy)
        """
        try:
            # Get average spectrum across time
            avg_spectrum = np.mean(magnitude_spectrum, axis=1)
            
            # Find fundamental frequency (strongest peak in low frequencies)
            # Look in range 50-500 Hz for double pedals
            low_freq_mask = (freqs >= 50) & (freqs <= 500)
            low_freq_spectrum = avg_spectrum[low_freq_mask]
            low_freq_freqs = freqs[low_freq_mask]
            
            if len(low_freq_spectrum) == 0:
                return 0.0
            
            # Find fundamental (strongest peak)
            fundamental_idx = np.argmax(low_freq_spectrum)
            fundamental_freq = low_freq_freqs[fundamental_idx]
            
            if fundamental_freq < 50:  # Too low, likely noise
                return 0.0
            
            # Extract first 10 harmonics
            harmonic_amplitudes = []
            for n in range(1, 11):  # n=1 is fundamental, n=2-10 are harmonics
                harmonic_freq = fundamental_freq * n
                if harmonic_freq >= freqs[-1]:  # Beyond Nyquist
                    break
                
                # Find closest frequency bin
                freq_idx = np.argmin(np.abs(freqs - harmonic_freq))
                if freq_idx < len(avg_spectrum):
                    amplitude = avg_spectrum[freq_idx]
                    harmonic_amplitudes.append(amplitude)
            
            if len(harmonic_amplitudes) < 3:  # Need at least a few harmonics
                return 0.0
            
            # Normalize by fundamental
            fundamental_amp = harmonic_amplitudes[0]
            if fundamental_amp < 1e-10:
                return 0.0
            
            normalized_harmonics = [amp / fundamental_amp for amp in harmonic_amplitudes[1:]]
            
            # Calculate expected sawtooth pattern: 1/n for nth harmonic
            # So 2nd harmonic should be ~0.5, 3rd ~0.33, 4th ~0.25, etc.
            expected_pattern = [1.0 / (n + 1) for n in range(1, len(normalized_harmonics) + 1)]
            
            # Compare actual to expected using correlation
            if len(normalized_harmonics) >= 3:
                # Use cosine similarity or correlation
                actual = np.array(normalized_harmonics[:min(8, len(normalized_harmonics))])
                expected = np.array(expected_pattern[:len(actual)])
                
                # Normalize
                actual_norm = actual / (np.linalg.norm(actual) + 1e-10)
                expected_norm = expected / (np.linalg.norm(expected) + 1e-10)
                
                # Cosine similarity (correlation)
                similarity = np.dot(actual_norm, expected_norm)
                
                # Also check that harmonics are present (not just fundamental)
                harmonic_strength = np.mean(normalized_harmonics[:5])  # Average of first 5 harmonics
                
                # Combine: similarity to pattern + presence of harmonics
                score = (similarity * 0.7 + min(1.0, harmonic_strength * 2) * 0.3)
                return float(max(0.0, min(1.0, score)))
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"    Error calculating sawtooth similarity: {e}")
            return 0.0
    
    def save_analysis(self, analysis: Dict[str, Any], output_path: str):
        """Save analysis results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"  Analysis saved to: {output_path}")


def get_audio_analyzer(sample_rate: int = 22050, hop_length: int = 512) -> AudioAnalyzer:
    """Get or create audio analyzer instance."""
    return AudioAnalyzer(sample_rate=sample_rate, hop_length=hop_length)
