"""Proprietary scoring/calibration rubric WITHHELD in this portfolio snapshot.

The feature-extraction layer (spectral / spit-buzz / tone / pitch features) is
retained to show the DSP approach; the exact thresholds, scoring weights, and
calibration distributions that encode the teaching rubric are not published.
See docs/portfolio-snapshot.md.
"""


class PlayedPassageToneAnalyzer:
    """Played-passage tone analyzer (the unified tone scorer).

    Approach (withheld): onset-segment a melodic passage -> per-note tone
    features -> score against a two-tier calibrated target -> per-note and
    aggregate results with feedback.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def analyze_played_passage(self, *args, **kwargs):
        raise NotImplementedError("Tone-analysis rubric withheld in portfolio snapshot.")
