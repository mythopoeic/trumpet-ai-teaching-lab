"""Unit tests for the shared percentile mapping (``audio.percentile``).

The value-to-percentile mapping the spit-buzz and tone scorers both use. Pure
Python, no numpy/librosa -- the reason it was extracted.

Run from ``teaching-engine/``:

    .venv/Scripts/python.exe -m pytest tests/unit/test_percentile.py -v
"""

from audio.percentile import value_to_percentile


# --- basic mapping ---------------------------------------------------------


def test_monotonic_endpoints_and_midpoint():
    refs = [1.0, 2.0, 3.0, 4.0]
    assert value_to_percentile(0.0, refs) == 0.0  # below all
    assert value_to_percentile(4.0, refs) == 100.0  # at/above all
    assert value_to_percentile(2.5, refs) == 50.0  # half at-or-below


def test_empty_distribution_is_neutral_50():
    assert value_to_percentile(123.0, []) == 50.0


def test_uses_bisect_right_semantics_for_ties():
    # bisect_right counts values <= value; a value equal to refs counts the
    # ties as "at or below" (right side).
    refs = [1.0, 2.0, 2.0, 2.0, 5.0]
    # 4 of 5 refs are <= 2.0 -> 80.0
    assert value_to_percentile(2.0, refs) == 80.0


def test_value_above_max_clamps_to_100():
    assert value_to_percentile(999.0, [1.0, 2.0]) == 100.0


def test_value_below_min_is_0():
    assert value_to_percentile(-5.0, [1.0, 2.0]) == 0.0


# --- inversion -------------------------------------------------------------


def test_inverted_flips_the_percentile():
    refs = [1.0, 2.0, 3.0, 4.0]
    # Non-inverted 2.5 -> 50.0; inverted -> 50.0 (symmetric midpoint).
    assert value_to_percentile(2.5, refs, inverted=True) == 50.0
    # A low raw value is GOOD when inverted: 0.0 maps to 0.0 normally,
    # flips to 100.0.
    assert value_to_percentile(0.0, refs, inverted=True) == 100.0
    # A high raw value is BAD when inverted: 4.0 maps to 100.0, flips to 0.0.
    assert value_to_percentile(4.0, refs, inverted=True) == 0.0


def test_inverted_empty_distribution_still_neutral():
    # Empty -> 50.0 before inversion; flipping 50.0 stays 50.0.
    assert value_to_percentile(1.0, [], inverted=True) == 50.0
