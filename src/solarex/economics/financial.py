"""Solar PV financial analysis — LCOE, NPV, IRR, payback."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SolarFinancialInputs:
    """Input parameters for solar PV financial analysis."""

    capacity_mw: float = 10.0
    capacity_factor: float = 0.20
    capex_per_kw: float = 1000.0      # $/kW
    opex_per_kw_yr: float = 15.0      # $/kW/year
    discount_rate: float = 0.08
    lifetime_years: int = 25
    electricity_price: float = 50.0    # $/MWh
    degradation_rate: float = 0.005    # /year


@dataclass
class SolarFinancialResults:
    """Output of solar PV financial analysis."""

    lcoe: float = 0.0             # $/MWh
    npv: float = 0.0              # $
    irr: float = 0.0              # fraction
    payback_years: float = 0.0
    annual_revenue: float = 0.0   # $/year (year 1)
    annual_opex: float = 0.0      # $/year
    total_generation_mwh: float = 0.0  # lifetime
    capex_total: float = 0.0      # $


def _crf(rate: float, years: int) -> float:
    """Capital Recovery Factor."""
    if rate <= 0:
        return 1.0 / max(years, 1)
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)


def _compute_irr(cash_flows: list[float], tol: float = 1e-6) -> float:
    """Internal Rate of Return via bisection."""
    lo, hi = -0.5, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        npv = sum(cf / (1 + mid) ** t for t, cf in enumerate(cash_flows))
        if abs(npv) < tol:
            return mid
        if npv > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def compute_pv_financials(
    inputs: SolarFinancialInputs,
) -> SolarFinancialResults:
    """Compute LCOE, NPV, IRR, and payback for a solar PV project."""
    cap_kw = inputs.capacity_mw * 1000.0
    capex_total = cap_kw * inputs.capex_per_kw
    annual_opex = cap_kw * inputs.opex_per_kw_yr

    # Year-1 generation
    annual_gen_yr1 = inputs.capacity_mw * inputs.capacity_factor * 8760.0  # MWh

    # LCOE via CRF
    crf = _crf(inputs.discount_rate, inputs.lifetime_years)
    annual_capex_equiv = capex_total * crf
    lcoe = (
        (annual_capex_equiv + annual_opex) / annual_gen_yr1
        if annual_gen_yr1 > 0
        else float("inf")
    )

    # NPV + total generation + payback
    total_gen = 0.0
    cumulative_cf = -capex_total
    payback = float(inputs.lifetime_years)
    payback_found = False

    cash_flows = [-capex_total]
    for t in range(1, inputs.lifetime_years + 1):
        deg = (1.0 - inputs.degradation_rate) ** (t - 1)
        gen_t = annual_gen_yr1 * deg
        total_gen += gen_t
        revenue_t = gen_t * inputs.electricity_price
        cf_t = revenue_t - annual_opex
        cash_flows.append(cf_t)

        cumulative_cf += cf_t
        if not payback_found and cumulative_cf >= 0:
            prev = cumulative_cf - cf_t
            frac = -prev / cf_t if cf_t != 0 else 0
            payback = (t - 1) + frac
            payback_found = True

    npv = sum(
        cf / (1 + inputs.discount_rate) ** t
        for t, cf in enumerate(cash_flows)
    )

    irr = _compute_irr(cash_flows) if len(cash_flows) > 1 else 0.0

    year1_revenue = annual_gen_yr1 * inputs.electricity_price

    return SolarFinancialResults(
        lcoe=lcoe,
        npv=npv,
        irr=irr,
        payback_years=payback,
        annual_revenue=year1_revenue,
        annual_opex=annual_opex,
        total_generation_mwh=total_gen,
        capex_total=capex_total,
    )
