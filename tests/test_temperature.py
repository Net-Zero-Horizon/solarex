"""Tests for solarex.core.temperature."""

import numpy as np
import pytest

from solarex.core.temperature import compute_temp_analysis


class TestTempAnalysis:
    def test_shape(self):
        ghi = np.full(100, 800.0)
        temp = np.full(100, 25.0)
        t_cell, derating = compute_temp_analysis(ghi, temp, 45.0)
        assert t_cell.shape == (100,)
        assert derating.shape == (100,)

    def test_cell_temp_higher_than_ambient(self):
        ghi = np.full(100, 800.0)
        temp = np.full(100, 25.0)
        t_cell, _ = compute_temp_analysis(ghi, temp, 45.0)
        assert np.all(t_cell > temp)

    def test_cell_temp_equals_ambient_at_night(self):
        ghi = np.zeros(100)
        temp = np.full(100, 15.0)
        t_cell, _ = compute_temp_analysis(ghi, temp, 45.0)
        np.testing.assert_allclose(t_cell, temp)

    def test_derating_at_stc(self):
        # At STC: Tcell=25, derating should be 1.0
        ghi = np.zeros(10)
        temp = np.full(10, 25.0)
        _, derating = compute_temp_analysis(ghi, temp, 45.0)
        np.testing.assert_allclose(derating, 1.0)

    def test_derating_decreases_with_high_temp(self):
        ghi = np.full(100, 800.0)
        temp_cold = np.full(100, 5.0)
        temp_hot = np.full(100, 40.0)
        _, der_cold = compute_temp_analysis(ghi, temp_cold, 45.0)
        _, der_hot = compute_temp_analysis(ghi, temp_hot, 45.0)
        assert der_cold.mean() > der_hot.mean()

    def test_custom_gamma(self):
        ghi = np.full(10, 500.0)
        temp = np.full(10, 30.0)
        _, der_default = compute_temp_analysis(ghi, temp, 45.0, gamma_pmax=-0.40)
        _, der_low = compute_temp_analysis(ghi, temp, 45.0, gamma_pmax=-0.20)
        # Lower gamma -> less derating
        assert der_low.mean() > der_default.mean()
