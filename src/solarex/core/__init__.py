"""Core solar PV computation modules."""

from .irradiance import (
    compute_peak_sun_hours,
    compute_clearness_index,
    compute_diurnal_irradiance,
    compute_monthly_irradiance,
    compute_performance_ratio,
)
from .temperature import compute_temp_analysis
from .shading import compute_gcr_shading_loss, compute_gcr_curve
from .bifacial import compute_bifacial_gain
from .capacity_factor import compute_solar_hourly_cf, normalize_to_8760

__all__ = [
    "compute_peak_sun_hours",
    "compute_clearness_index",
    "compute_diurnal_irradiance",
    "compute_monthly_irradiance",
    "compute_performance_ratio",
    "compute_temp_analysis",
    "compute_gcr_shading_loss",
    "compute_gcr_curve",
    "compute_bifacial_gain",
    "compute_solar_hourly_cf",
    "normalize_to_8760",
]
