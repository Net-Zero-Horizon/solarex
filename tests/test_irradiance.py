"""Tests for solarex.core.irradiance."""

import numpy as np
import pytest

from solarex.core.irradiance import (
    compute_peak_sun_hours,
    compute_performance_ratio,
    compute_clearness_index,
    compute_diurnal_irradiance,
    compute_monthly_irradiance,
)


class TestPeakSunHours:
    def test_constant_irradiance(self):
        # 500 W/m2 for all hours -> PSH = 500*8760/1000/365 = 12.0 h/day
        ghi = np.full(8760, 500.0)
        psh = compute_peak_sun_hours(ghi)
        assert psh == pytest.approx(12.0, rel=0.01)

    def test_zero_irradiance(self):
        ghi = np.zeros(8760)
        psh = compute_peak_sun_hours(ghi)
        assert psh == 0.0

    def test_realistic_range(self):
        # Typical: 4-6 PSH for good solar site
        rng = np.random.default_rng(42)
        # Simulate day/night: 12h sun at ~600 W/m2 avg
        ghi = np.zeros(8760)
        for d in range(365):
            for h in range(6, 18):
                ghi[d * 24 + h] = rng.uniform(200, 900)
        psh = compute_peak_sun_hours(ghi)
        assert 2.0 < psh < 8.0


class TestPerformanceRatio:
    def test_stc_conditions(self):
        # At STC (25C, 1000 W/m2): PR should be ~1.0
        ghi = np.full(100, 1000.0)
        temp = np.full(100, 25.0)
        pr = compute_performance_ratio(ghi, temp, 0.20, -0.40, 45.0)
        # At 25C, temp factor = 1.0 + (-0.40/100)*(Tcell-25)
        # Tcell = 25 + (45-20)/800*1000 = 56.25
        # factor = 1.0 + (-0.004)*(56.25-25) = 0.875
        assert 0.8 < pr < 1.0

    def test_zero_irradiance(self):
        ghi = np.zeros(100)
        temp = np.full(100, 25.0)
        pr = compute_performance_ratio(ghi, temp, 0.20, -0.40, 45.0)
        assert pr == 0.0

    def test_cold_climate_better(self):
        ghi = np.full(100, 800.0)
        temp_cold = np.full(100, 0.0)
        temp_hot = np.full(100, 40.0)
        pr_cold = compute_performance_ratio(ghi, temp_cold, 0.20, -0.40, 45.0)
        pr_hot = compute_performance_ratio(ghi, temp_hot, 0.20, -0.40, 45.0)
        assert pr_cold > pr_hot


class TestClearnessIndex:
    def test_zero_data(self):
        kt = compute_clearness_index(np.array([]), 30.0, [])
        assert kt == 0.0

    def test_range(self):
        # Generate synthetic hourly data for one year
        rng = np.random.default_rng(42)
        ghi = np.zeros(8760)
        timestamps = []
        for d in range(365):
            for h in range(24):
                idx = d * 24 + h
                if 6 <= h <= 18:
                    ghi[idx] = rng.uniform(0, 800)
                timestamps.append(f"2022-{1 + d // 30:02d}-{1 + d % 30:02d}T{h:02d}:00")
        kt = compute_clearness_index(ghi, 30.0, timestamps)
        assert 0.0 <= kt <= 1.0


class TestDiurnalIrradiance:
    def test_shape(self):
        ghi = np.zeros(48)
        timestamps = [f"2022-01-01T{h:02d}:00" for h in range(24)]
        timestamps += [f"2022-01-02T{h:02d}:00" for h in range(24)]
        hours, means = compute_diurnal_irradiance(ghi, timestamps)
        assert len(hours) == 24
        assert len(means) == 24

    def test_daytime_peak(self):
        ghi = np.zeros(48)
        timestamps = []
        for d in range(2):
            date = f"2022-01-0{d+1}"
            for h in range(24):
                timestamps.append(f"{date}T{h:02d}:00")
                if 10 <= h <= 14:
                    ghi[d * 24 + h] = 800.0
        hours, means = compute_diurnal_irradiance(ghi, timestamps)
        assert means[12] > means[0]  # noon > midnight


class TestMonthlyIrradiance:
    def test_shape(self):
        ghi = np.full(8760, 500.0)
        timestamps = []
        from datetime import datetime, timedelta
        dt = datetime(2022, 1, 1)
        for _ in range(8760):
            timestamps.append(dt.isoformat())
            dt += timedelta(hours=1)
        months, totals = compute_monthly_irradiance(ghi, timestamps)
        assert len(months) == 12
        assert len(totals) == 12
        assert all(t > 0 for t in totals)
