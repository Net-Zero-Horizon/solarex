"""Tests for solarex.core.capacity_factor (offline parts)."""

import numpy as np
import pytest

from solarex.core.capacity_factor import normalize_to_8760, _irradiance_to_hourly_cf


class TestNormalizeTo8760:
    def test_exact_8760(self):
        cf = np.ones(8760)
        result = normalize_to_8760(cf)
        assert len(result) == 8760

    def test_leap_year_8784(self):
        cf = np.ones(8784)
        result = normalize_to_8760(cf)
        assert len(result) == 8760

    def test_short_padded(self):
        cf = np.ones(100)
        result = normalize_to_8760(cf)
        assert len(result) == 8760
        assert result[99] == 1.0
        assert result[100] == 0.0  # padded with zeros

    def test_too_long_trimmed(self):
        cf = np.ones(9000)
        result = normalize_to_8760(cf)
        assert len(result) == 8760


class TestIrradianceToCF:
    def test_stc_conditions(self):
        # At 1000 W/m2, 25C ambient -> CF should be close to 1
        ghi = np.array([1000.0])
        temp = np.array([25.0])
        cf = _irradiance_to_hourly_cf(ghi, temp, 0.20, -0.40, 45.0)
        # T_cell = 25 + (45-20)/800*1000 = 56.25
        # temp_factor = 1 + (-0.40/100)*(56.25-25) = 0.875
        # CF = 1.0 * 0.875 = 0.875
        assert cf[0] == pytest.approx(0.875, abs=0.01)

    def test_night_zero(self):
        ghi = np.array([0.0])
        temp = np.array([15.0])
        cf = _irradiance_to_hourly_cf(ghi, temp, 0.20, -0.40, 45.0)
        assert cf[0] == 0.0

    def test_clipped_to_01(self):
        ghi = np.array([1500.0, 0.0])
        temp = np.array([5.0, 5.0])
        cf = _irradiance_to_hourly_cf(ghi, temp, 0.20, -0.40, 45.0)
        assert np.all(cf >= 0.0)
        assert np.all(cf <= 1.0)

    def test_shape(self):
        ghi = np.random.default_rng(42).uniform(0, 1000, 8760)
        temp = np.random.default_rng(42).uniform(0, 35, 8760)
        cf = _irradiance_to_hourly_cf(ghi, temp, 0.20, -0.40, 45.0)
        assert cf.shape == (8760,)
