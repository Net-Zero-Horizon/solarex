"""Bifacial PV energy gain estimation using simplified view-factor model."""

from __future__ import annotations

import math


def compute_bifacial_gain(
    albedo: float,
    gcr: float,
    module_height: float,
    tilt: float,
    bifaciality: float = 0.70,
) -> float:
    """Estimate bifacial energy gain fraction.

    Uses simplified view-factor model for rear-side irradiance.

    Parameters
    ----------
    albedo : float
        Ground albedo (0-1). Typical: 0.25 grass, 0.60 sand, 0.80 snow.
    gcr : float
        Ground Coverage Ratio.
    module_height : float
        Module height in meters.
    tilt : float
        Module tilt in degrees.
    bifaciality : float
        Module bifaciality factor (typical 0.65-0.80).

    Returns
    -------
    float
        Bifacial gain fraction (e.g. 0.10 = 10% more energy).
    """
    if albedo <= 0 or gcr <= 0 or bifaciality <= 0:
        return 0.0

    tilt_rad = math.radians(tilt)

    # Clearance height (bottom of module above ground)
    clearance = max(0.3, module_height * math.sin(tilt_rad) * 0.3)

    # View factor from ground to rear of module (simplified)
    row_pitch = module_height / gcr if gcr > 0 else 10.0
    module_footprint = module_height * math.cos(tilt_rad)
    open_fraction = max(0.0, 1.0 - module_footprint / row_pitch)

    # View factor approximation (0-1)
    vf = open_fraction * min(1.0, clearance / module_height)

    # Rear irradiance fraction = albedo x view_factor
    rear_fraction = albedo * vf

    # Bifacial gain
    gain = rear_fraction * bifaciality

    return min(0.50, gain)  # Cap at 50% gain
