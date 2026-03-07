"""
SolareX -- Solar Resource eXchange

A Python library for solar PV resource assessment, irradiance analysis,
temperature modeling, shading/bifacial analysis, MCDA site scoring, and
capacity factor computation.

Modules:
- solarex.core: Irradiance, temperature, shading, bifacial, capacity factor
- solarex.data: Open-Meteo, NASA POWER, ERA5, terrain, LULC, module DB
- solarex.economics: Financial analysis and LCOE sensitivity
- solarex.analysis: MCDA engine and grid evaluation
- solarex.regional: Regional analyzer and zone generation
"""

__version__ = "0.1.0"
__author__ = "SolareX Development Team"

from .core.irradiance import (
    compute_peak_sun_hours,
    compute_clearness_index,
    compute_diurnal_irradiance,
    compute_monthly_irradiance,
    compute_performance_ratio,
)
from .core.temperature import compute_temp_analysis
from .core.shading import compute_gcr_shading_loss, compute_gcr_curve
from .core.bifacial import compute_bifacial_gain

__all__ = [
    "__version__",
    # Irradiance
    "compute_peak_sun_hours",
    "compute_clearness_index",
    "compute_diurnal_irradiance",
    "compute_monthly_irradiance",
    "compute_performance_ratio",
    # Temperature
    "compute_temp_analysis",
    # Shading
    "compute_gcr_shading_loss",
    "compute_gcr_curve",
    # Bifacial
    "compute_bifacial_gain",
]


def __getattr__(name):
    """Lazy import for heavy modules."""
    if name == "compute_solar_hourly_cf":
        from .core.capacity_factor import compute_solar_hourly_cf
        return compute_solar_hourly_cf
    if name == "SolarPVAnalyzer":
        from .regional.analyzer import SolarPVAnalyzer
        return SolarPVAnalyzer
    if name in ("compute_pv_financials", "SolarFinancialInputs", "SolarFinancialResults"):
        from .economics import financial
        return getattr(financial, name)
    if name == "compute_mcda_scores":
        from .analysis.mcda import compute_mcda_scores
        return compute_mcda_scores
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
