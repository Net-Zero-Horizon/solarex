"""GCR inter-row shading analysis for ground-mount solar arrays."""

from __future__ import annotations

import math
import os


def compute_gcr_shading_loss(
    latitude: float,
    tilt: float,
    gcr: float,
    module_height: float = 2.0,
) -> float:
    """Estimate inter-row shading loss fraction for a ground-mount array.

    Uses winter solstice solar geometry for worst-case analysis.

    Parameters
    ----------
    latitude : float
        Site latitude in degrees.
    tilt : float
        Module tilt angle in degrees from horizontal.
    gcr : float
        Ground Coverage Ratio (0-1). GCR = module_width / row_pitch.
    module_height : float
        Module height (collector width) in meters.

    Returns
    -------
    float
        Shading loss fraction (0-1). 0 = no shading, 1 = fully shaded.
    """
    if gcr <= 0 or gcr > 1:
        return 0.0

    tilt_rad = math.radians(tilt)
    lat_rad = math.radians(abs(latitude))

    # Winter solstice declination (-23.45 for NH, +23.45 for SH)
    decl_rad = math.radians(-23.45) if latitude >= 0 else math.radians(23.45)

    # Solar altitude at noon on winter solstice
    sin_alt = (
        math.sin(lat_rad) * math.sin(decl_rad)
        + math.cos(lat_rad) * math.cos(decl_rad)
    )
    solar_alt = math.asin(max(-1.0, min(1.0, sin_alt)))

    if solar_alt <= 0:
        return 1.0

    # Shadow length cast by back edge of module
    module_top_height = module_height * math.sin(tilt_rad)
    shadow_length = module_top_height / math.tan(solar_alt)

    # Row pitch from GCR
    row_pitch = module_height / gcr if gcr > 0 else float("inf")

    # Module horizontal footprint
    module_footprint = module_height * math.cos(tilt_rad)

    # Available gap between rows
    gap = row_pitch - module_footprint

    if gap <= 0:
        return min(1.0, shadow_length / module_footprint) if module_footprint > 0 else 0.0

    if shadow_length <= gap:
        return 0.0

    shaded_portion = shadow_length - gap
    shading_loss = min(1.0, shaded_portion / module_footprint) if module_footprint > 0 else 0.0

    # Weight by approximate fraction of day affected
    weighted_loss = shading_loss * 0.30
    return min(1.0, weighted_loss)


def compute_gcr_curve(
    latitude: float,
    tilt: float,
    gcrs: list[float] | None = None,
    module_height: float = 2.0,
    max_workers: int = 0,
) -> tuple[list[float], list[float]]:
    """Compute shading loss vs GCR curve.

    Returns (gcr_values, loss_fractions).
    """
    from concurrent.futures import ThreadPoolExecutor

    if gcrs is None:
        gcrs = [round(0.15 + 0.05 * i, 2) for i in range(15)]  # 0.15 to 0.85

    def _eval(g):
        return compute_gcr_shading_loss(latitude, tilt, g, module_height)

    workers = max_workers if max_workers > 0 else (os.cpu_count() or 4)
    n_workers = min(workers, len(gcrs))

    if n_workers <= 1 or len(gcrs) <= 3:
        losses = [_eval(g) for g in gcrs]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            losses = list(pool.map(_eval, gcrs))

    return gcrs, losses
