"""ERA5 solar irradiance data via atlite library.

Requires: pip install solarex[era5]
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_cf_atlite(
    south: float,
    west: float,
    north: float,
    east: float,
    year: int,
    module_efficiency: float = 0.20,
    orientation: str = "latitude_optimal",
    tilt: float = 0.0,
    azimuth: float = 180.0,
    tracking: str = "none",
    on_progress: Any = None,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Download ERA5 data via atlite and compute mean PV capacity factors.

    Parameters
    ----------
    south, west, north, east : float
        Bounding box (WGS84).
    year : int
        Calendar year.
    module_efficiency : float
        Module STC efficiency.
    orientation : str
        "latitude_optimal" or "custom".
    tilt, azimuth : float
        Custom orientation angles (degrees).
    tracking : str
        "none", "horizontal", "vertical", "dual".
    on_progress : callable or None
        Callback ``(pct: int, msg: str)``.

    Returns
    -------
    tuple
        (mean_cf_2d, ghi_annual_2d, lats_1d, lons_1d)
    """
    try:
        import atlite
    except ImportError:
        logger.error("atlite is not installed. Install with: pip install solarex[era5]")
        return np.array([]), None, np.array([]), np.array([])

    import tempfile
    from pathlib import Path

    tmpdir = Path(tempfile.mkdtemp(prefix="solarex_era5_"))
    cutout_path = tmpdir / "solar_pv_cutout.nc"

    if on_progress:
        on_progress(5, "Downloading ERA5 irradiance data via CDS...")

    cutout = atlite.Cutout(
        path=cutout_path,
        module="era5",
        x=slice(west, east),
        y=slice(south, north),
        time=str(year),
    )
    cutout.prepare()

    if on_progress:
        on_progress(20, "Computing solar PV capacity factors...")

    panel_config = {
        "model": "huld",
        "efficiency": module_efficiency,
        "c_temp_amb": 1.0,
        "c_temp_irrad": 0.035,
        "r_tmod": 298.0,
        "r_tamb": 293.0,
        "r_irradiance": 1000.0,
        "inverter_efficiency": 0.96,
    }

    if orientation == "latitude_optimal":
        mid_lat = (south + north) / 2.0
        orient = {
            "slope": abs(mid_lat),
            "azimuth": 180.0 if mid_lat >= 0 else 0.0,
        }
    else:
        orient = {"slope": tilt, "azimuth": azimuth}

    pv_kwargs: dict = {
        "panel": panel_config,
        "orientation": orient,
        "capacity_factor_timeseries": True,
    }
    if tracking in ("horizontal", "vertical", "dual"):
        pv_kwargs["tracking"] = tracking

    cf_ts = cutout.pv(**pv_kwargs)

    mean_cf = cf_ts.mean(dim="time").values
    lats = cf_ts.coords["y"].values
    lons = cf_ts.coords["x"].values

    ghi_annual = None
    try:
        influx = cutout.data["influx_direct"] + cutout.data["influx_diffuse"]
        ghi_annual = influx.sum(dim="time").values / 1000.0
    except Exception:
        logger.debug("Could not compute GHI from cutout")

    return mean_cf, ghi_annual, lats, lons
