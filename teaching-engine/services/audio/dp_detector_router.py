"""Proprietary scoring/calibration rubric WITHHELD in this portfolio snapshot.

The feature-extraction layer (spectral / spit-buzz / tone / pitch features) is
retained to show the DSP approach; the exact thresholds, scoring weights, and
calibration distributions that encode the teaching rubric are not published.
See docs/portfolio-snapshot.md.
"""


def detect_attempts(*args, **kwargs):
    """Segment a recording into attempts and run the tiered double-pedal detector.

    Approach (withheld): per-attempt windows -> Tier A register gate, Tier B
    pitch/cents decode. Exact thresholds and calibration are not published.
    """
    raise NotImplementedError("Double-pedal detector rubric withheld in portfolio snapshot.")
