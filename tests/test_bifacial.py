"""Tests for solarex.core.bifacial."""

import pytest

from solarex.core.bifacial import compute_bifacial_gain


class TestBifacialGain:
    def test_zero_albedo(self):
        assert compute_bifacial_gain(0.0, 0.4, 2.0, 25.0) == 0.0

    def test_zero_gcr(self):
        assert compute_bifacial_gain(0.25, 0.0, 2.0, 25.0) == 0.0

    def test_positive_gain(self):
        gain = compute_bifacial_gain(0.25, 0.4, 2.0, 25.0)
        assert gain > 0.0

    def test_higher_albedo_more_gain(self):
        g_grass = compute_bifacial_gain(0.25, 0.4, 2.0, 25.0)
        g_snow = compute_bifacial_gain(0.80, 0.4, 2.0, 25.0)
        assert g_snow > g_grass

    def test_capped_at_50_percent(self):
        # Even extreme albedo should be capped
        gain = compute_bifacial_gain(1.0, 0.2, 5.0, 45.0, bifaciality=1.0)
        assert gain <= 0.50

    def test_range(self):
        gain = compute_bifacial_gain(0.25, 0.4, 2.0, 25.0, 0.70)
        assert 0.0 <= gain <= 0.50
