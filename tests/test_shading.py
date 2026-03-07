"""Tests for solarex.core.shading."""

import pytest

from solarex.core.shading import compute_gcr_shading_loss, compute_gcr_curve


class TestGCRShadingLoss:
    def test_zero_gcr(self):
        assert compute_gcr_shading_loss(30.0, 25.0, 0.0) == 0.0

    def test_invalid_gcr(self):
        assert compute_gcr_shading_loss(30.0, 25.0, -0.1) == 0.0
        assert compute_gcr_shading_loss(30.0, 25.0, 1.5) == 0.0

    def test_low_gcr_no_shading(self):
        # Very low GCR at moderate latitude -> minimal or no shading
        loss = compute_gcr_shading_loss(30.0, 25.0, 0.2)
        assert loss < 0.1

    def test_high_gcr_more_shading(self):
        loss_low = compute_gcr_shading_loss(30.0, 25.0, 0.3)
        loss_high = compute_gcr_shading_loss(30.0, 25.0, 0.7)
        assert loss_high >= loss_low

    def test_higher_latitude_more_shading(self):
        loss_low_lat = compute_gcr_shading_loss(20.0, 25.0, 0.5)
        loss_high_lat = compute_gcr_shading_loss(55.0, 25.0, 0.5)
        assert loss_high_lat >= loss_low_lat

    def test_range(self):
        loss = compute_gcr_shading_loss(40.0, 30.0, 0.5)
        assert 0.0 <= loss <= 1.0


class TestGCRCurve:
    def test_default_gcrs(self):
        gcrs, losses = compute_gcr_curve(30.0, 25.0)
        assert len(gcrs) == 15
        assert len(losses) == 15
        assert all(0.0 <= l <= 1.0 for l in losses)

    def test_custom_gcrs(self):
        gcrs, losses = compute_gcr_curve(30.0, 25.0, gcrs=[0.2, 0.4, 0.6])
        assert len(gcrs) == 3
        assert len(losses) == 3

    def test_monotonic_trend(self):
        _, losses = compute_gcr_curve(40.0, 30.0, gcrs=[0.2, 0.3, 0.5, 0.7])
        # Losses should generally increase with GCR (may have flat regions)
        assert losses[-1] >= losses[0]
