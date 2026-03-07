"""Tests for solarex.economics.financial."""

import pytest

from solarex.economics.financial import (
    SolarFinancialInputs,
    SolarFinancialResults,
    compute_pv_financials,
)


class TestFinancials:
    def test_basic(self):
        inputs = SolarFinancialInputs()
        result = compute_pv_financials(inputs)
        assert isinstance(result, SolarFinancialResults)
        assert result.lcoe > 0
        assert result.capex_total > 0
        assert result.total_generation_mwh > 0

    def test_lcoe_increases_with_capex(self):
        base = compute_pv_financials(SolarFinancialInputs(capex_per_kw=800))
        high = compute_pv_financials(SolarFinancialInputs(capex_per_kw=1600))
        assert high.lcoe > base.lcoe

    def test_lcoe_decreases_with_cf(self):
        low_cf = compute_pv_financials(SolarFinancialInputs(capacity_factor=0.15))
        high_cf = compute_pv_financials(SolarFinancialInputs(capacity_factor=0.25))
        assert low_cf.lcoe > high_cf.lcoe

    def test_npv_positive_for_cheap_solar(self):
        inputs = SolarFinancialInputs(
            capacity_factor=0.25,
            capex_per_kw=600,
            electricity_price=80,
        )
        result = compute_pv_financials(inputs)
        assert result.npv > 0
        assert result.irr > inputs.discount_rate

    def test_degradation_reduces_generation(self):
        no_deg = compute_pv_financials(SolarFinancialInputs(degradation_rate=0.0))
        with_deg = compute_pv_financials(SolarFinancialInputs(degradation_rate=0.01))
        assert no_deg.total_generation_mwh > with_deg.total_generation_mwh
