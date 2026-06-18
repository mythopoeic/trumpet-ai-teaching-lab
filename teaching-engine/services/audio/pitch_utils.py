"""Pitch and ladder utilities for double-pedal Tier B decoding.

This module is the single source of truth for the seven-rung Trumpet Yoga
double-pedal ladder (``YOGA_LADDER_WRITTEN_BB``) and a small set of
note-name <-> frequency helpers shared by the Tier B decoder and any
future caller that needs to translate between concert and written
notation.

Layout
------
- ``YOGA_LADDER_WRITTEN_BB``: the ordered list of seven Yoga ladder rungs
  (top down: written C2 .. F#1). Each entry includes the written name,
  the concert name, and the concert fundamental in Hz at A4 = 440 Hz
  equal temperament. The Hz values are what audio F0 estimators see;
  cents distances in ``dp_tier_b`` are computed against these numbers.
- ``note_name_to_freq(note_name)``: parse a fully-qualified scientific
  pitch name (e.g. ``"Bb1"``, ``"F#1"``, ``"C2"``) and return its
  frequency in Hz at A4 = 440.
- ``freq_to_note_name(freq_hz)``: inverse mapping. Returns the nearest
  semitone name (with octave) and the signed cents offset relative to
  it.
- ``concert_to_written(concert_note, instrument_key='Bb')``: shift a
  concert note up a major 2nd for Bb-trumpet notation. ``instrument_key``
  is reserved for future use; only ``'Bb'`` is implemented.
- ``written_to_concert(written_note, instrument_key='Bb')``: inverse.
- ``cents_distance(f, f_ref)``: ``1200 * log2(f / f_ref)``. Signed.

Conventions
-----------
- Internal note ordering uses sharps (C, C#, D, D#, ...), so flats are
  normalized via the enharmonic table before any frequency math.
- ``YOGA_LADDER_WRITTEN_BB`` keeps the spelling the source PRD prescribes
  (mixed sharps and flats: C, B, Bb, A, Ab, G, F#) so its display labels
  match the curriculum drills.
- All pitches in this module are equal-tempered. The Yoga ladder spans
  approximately 41-58 Hz (concert E1 .. concert Bb1), one octave below
  the standard trumpet pedal range.

Python 3.9. ASCII-only.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


__all__ = [
    'YOGA_LADDER_WRITTEN_BB',
    'note_name_to_freq',
    'freq_to_note_name',
    'concert_to_written',
    'written_to_concert',
    'cents_distance',
]


# Enharmonic normalization: flats -> sharps. Used by the parser only.
_ENHARMONIC_TO_SHARP: Dict[str, str] = {
    'Cb': 'B',   # Cb in octave N is the same pitch as B in octave (N-1); handled in parser.
    'Db': 'C#',
    'Eb': 'D#',
    'Fb': 'E',
    'Gb': 'F#',
    'Ab': 'G#',
    'Bb': 'A#',
}

# Sharp-spelling pitch class indices (semitones above C).
_SHARP_NAMES: List[str] = [
    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
]

# Reverse lookup for parsing.
_SHARP_INDEX: Dict[str, int] = {name: i for i, name in enumerate(_SHARP_NAMES)}


def _split_note_name(note_name: str) -> Tuple[str, int]:
    """Split a scientific pitch name into (letter+accidental, octave).

    Accepts ``"C2"``, ``"Bb1"``, ``"F#1"``, ``"G#1"``, etc. Octave must
    be a non-negative integer. Raises ``ValueError`` on malformed input.
    """
    s = note_name.strip()
    if not s:
        raise ValueError(f"empty note name")
    # Find where digits start.
    i = 0
    while i < len(s) and not s[i].isdigit():
        i += 1
    if i == 0 or i == len(s):
        raise ValueError(f"note name missing letter or octave: {note_name!r}")
    pitch_part = s[:i]
    octave_part = s[i:]
    try:
        octave = int(octave_part)
    except ValueError as exc:
        raise ValueError(f"invalid octave in note name {note_name!r}: {exc}") from exc
    return pitch_part, octave


def _pitch_to_semitones(pitch: str, octave: int) -> int:
    """Translate a (pitch, octave) pair to absolute semitones from C0.

    Handles enharmonic spellings (Bb, Db, Eb, ...) by normalizing to the
    sharp-spelling pitch class. Cb in octave N is the same pitch as B in
    octave (N-1) and is handled with an explicit octave decrement.
    """
    raw = pitch.strip()
    if not raw:
        raise ValueError(f"empty pitch token")
    octave_offset = 0
    if raw == 'Cb':
        sharp_name = 'B'
        octave_offset = -1
    elif raw == 'B#':
        sharp_name = 'C'
        octave_offset = 1
    elif raw in _ENHARMONIC_TO_SHARP:
        sharp_name = _ENHARMONIC_TO_SHARP[raw]
    else:
        sharp_name = raw
    if sharp_name not in _SHARP_INDEX:
        raise ValueError(f"unknown pitch token: {pitch!r}")
    return (octave + octave_offset) * 12 + _SHARP_INDEX[sharp_name]


def _semitones_to_freq(semitones_from_c0: int) -> float:
    """Convert semitones-from-C0 to frequency in Hz at A4 = 440.

    A4 = 440 Hz sits at semitone index 4 * 12 + 9 = 57.
    """
    a4_index = 4 * 12 + 9
    return 440.0 * (2.0 ** ((semitones_from_c0 - a4_index) / 12.0))


def note_name_to_freq(note_name: str) -> float:
    """Parse a fully-qualified scientific pitch name and return Hz.

    Examples
    --------
    >>> note_name_to_freq('A4')
    440.0
    >>> round(note_name_to_freq('Bb1'), 3)
    58.27
    """
    pitch, octave = _split_note_name(note_name)
    return _semitones_to_freq(_pitch_to_semitones(pitch, octave))


def freq_to_note_name(freq_hz: float) -> Tuple[str, float]:
    """Return (nearest semitone name with octave, signed cents offset).

    Cents offset is positive when ``freq_hz`` is sharp of the named
    semitone, negative when flat. Returns ``('?', 0.0)`` for non-positive
    or NaN inputs (so callers do not have to special-case malformed F0).
    """
    if not isinstance(freq_hz, (int, float)) or freq_hz != freq_hz or freq_hz <= 0.0:
        return ('?', 0.0)
    a4_index = 4 * 12 + 9
    semitone_float = a4_index + 12.0 * math.log2(float(freq_hz) / 440.0)
    nearest = int(round(semitone_float))
    cents = (semitone_float - nearest) * 100.0
    octave = nearest // 12
    name = _SHARP_NAMES[nearest % 12]
    return (f"{name}{octave}", cents)


def concert_to_written(concert_note: str, instrument_key: str = 'Bb') -> str:
    """Translate a concert pitch to its written equivalent for the given key.

    For Bb instruments, the written pitch sounds a major 2nd lower than
    notated -- so written = concert + 2 semitones. Returns the
    sharp-spelling output name (e.g. ``concert_to_written('Bb1') == 'C2'``,
    ``concert_to_written('A1') == 'B1'``, ``concert_to_written('E1') == 'F#1'``).
    """
    if instrument_key != 'Bb':
        raise ValueError(
            f"only instrument_key='Bb' is supported; got {instrument_key!r}"
        )
    pitch, octave = _split_note_name(concert_note)
    semitones = _pitch_to_semitones(pitch, octave) + 2
    new_octave = semitones // 12
    name = _SHARP_NAMES[semitones % 12]
    return f"{name}{new_octave}"


def written_to_concert(written_note: str, instrument_key: str = 'Bb') -> str:
    """Translate a written (Bb-trumpet) pitch back to concert.

    Inverse of :func:`concert_to_written`. Written sounds a major 2nd
    LOWER, so concert = written - 2 semitones.
    """
    if instrument_key != 'Bb':
        raise ValueError(
            f"only instrument_key='Bb' is supported; got {instrument_key!r}"
        )
    pitch, octave = _split_note_name(written_note)
    semitones = _pitch_to_semitones(pitch, octave) - 2
    new_octave = semitones // 12
    name = _SHARP_NAMES[semitones % 12]
    return f"{name}{new_octave}"


def cents_distance(f: float, f_ref: float) -> float:
    """Signed cents from ``f_ref`` to ``f``: ``1200 * log2(f / f_ref)``.

    Positive when ``f`` is sharp of ``f_ref``; negative when flat.
    Raises ``ValueError`` if either argument is non-positive.
    """
    if f <= 0.0 or f_ref <= 0.0:
        raise ValueError(
            f"cents_distance requires positive frequencies; got f={f}, f_ref={f_ref}"
        )
    return 1200.0 * math.log2(f / f_ref)


def _build_yoga_ladder() -> List[Dict[str, Any]]:
    """Compose the seven-rung Yoga ladder once at import time.

    Source PRD: ``tasks/prd-yoga-double-pedals.md`` US-003 fixes the
    written-name list as ``['C', 'B', 'Bb', 'A', 'Ab', 'G', 'F#']``,
    descending. With written = concert + major 2nd, the concert
    fundamentals fall in the 41-58 Hz band (one octave below the
    trumpet's bell range). The exact written and concert spellings here
    follow the source PRD's mixed sharps/flats convention so display
    labels match drill text verbatim.
    """
    rungs: List[Dict[str, Any]] = []
    # (written_name_with_octave, concert_name_with_octave) pairs, top-down.
    pairs: List[Tuple[str, str]] = [
        ('C2',  'Bb1'),
        ('B1',  'A1'),
        ('Bb1', 'Ab1'),
        ('A1',  'G1'),
        ('Ab1', 'Gb1'),
        ('G1',  'F1'),
        ('F#1', 'E1'),
    ]
    for written, concert in pairs:
        rungs.append({
            'written': written,
            'concert': concert,
            'concert_hz': note_name_to_freq(concert),
        })
    return rungs


# The seven-rung Trumpet Yoga double-pedal ladder for Bb trumpet, top down.
YOGA_LADDER_WRITTEN_BB: List[Dict[str, Any]] = _build_yoga_ladder()


def _resolve_concert_target(
    target_note_concert: Optional[str],
) -> Tuple[Optional[str], Optional[float]]:
    """Resolve a concert-note target string to (written_rung_name, concert_hz).

    Accepts either fully-qualified scientific pitch (``'Bb1'``) or a bare
    letter (``'Bb'``); the bare form is matched against the ladder by
    stripping octave digits from each rung's concert name. Returns
    ``(None, None)`` when input is empty or no rung matches.
    """
    if not target_note_concert:
        return (None, None)
    raw = target_note_concert.strip()
    if not raw:
        return (None, None)
    # Exact match against the ladder's concert-with-octave spelling.
    for rung in YOGA_LADDER_WRITTEN_BB:
        if rung['concert'] == raw:
            return (rung['written'], float(rung['concert_hz']))
    # Bare-letter fallback: strip octave digits from the rung concert name.
    for rung in YOGA_LADDER_WRITTEN_BB:
        rung_letter = ''.join(c for c in rung['concert'] if not c.isdigit())
        if rung_letter == raw:
            return (rung['written'], float(rung['concert_hz']))
    # Last resort: parse as full scientific pitch (caller asked for an
    # off-ladder note such as a tuning reference).
    try:
        hz = note_name_to_freq(raw)
    except ValueError:
        return (None, None)
    try:
        written = concert_to_written(raw)
    except ValueError:
        return (None, hz)
    return (written, hz)
