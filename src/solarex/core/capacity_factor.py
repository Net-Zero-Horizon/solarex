"""Solar PV hourly capacity factor computation.

Computes hourly CF time series from weather reanalysis data using the
NOCT cell temperature model and temperature-corrected power output.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_HOURS_PER_YEAR = 8760
_FEB29_START = 24 * (31 + 28)  # hour index where Feb 29 starts (1416)
_FEB29_END = _FEB29_START + 24  # hour index where Feb 29 ends (1440)


def compute_solar_hourly_cf(
    lat: float,
    lon: float,
    year: int,
    data_source: str = "open_meteo",
    efficiency: float = 0.20,
    gamma_pmax: float = -0.40,
    t_noct: float = 45.0,
    tilt: Optional[float] = None,
    azimuth: float = 180.0,
    tracking: str = "none",
) -> np.ndarray:
    """Compute hourly PV capacity factor for a single location.

    Parameters
    ----------
    lat, lon : float
        Geographic coordinates (WGS84).
    year : int
        Calendar year for weather data.
    data_source : str
        One of ``"open_meteo"``, ``"nasa_power"``, ``"era5_atlite"``.
    efficiency : float
        Module STC efficiency (0-1), default 0.20.
    gamma_pmax : float
        Temperature coefficient of power (%/C), typically negative.
    t_noct : float
        Nominal operating cell temperature (C).
    tilt : float or None
        Panel tilt in degrees.  None = latitude-optimal.
    azimuth : float
        Panel azimuth in degrees (180 = south-facing).
    tracking : str
        Tracking mode: ``"none"``, ``"horizontal"``, ``"vertical"``, ``"dual"``.

    Returns
    -------
    np.ndarray
        Hourly capacity factors with shape ``(8760,)`` and values in [0, 1].
    """
    if data_source == "era5_atlite":
        return _compute_atlite(
            lat, lon, year, efficiency, gamma_pmax, t_noct,
            tilt, azimuth, tracking,
        )
    elif data_source == "nasa_power":
        from ..data.nasa_power import fetch_nasa_power_solar
        result = fetch_nasa_power_solar(lat, lon, year)
    else:
        from ..data.open_meteo import fetch_open_meteo_solar
        result = fetch_open_meteo_solar(lat, lon, year)

    if result is None:
        logger.warning(
            "Failed to fetch solar data for (%.4f, %.4f) year %d, "
            "returning zeros.",
            lat, lon, year,
        )
        return np.zeros(_HOURS_PER_YEAR)

    ghi, temp = result[0], result[1]
    cf = _irradiance_to_hourly_cf(ghi, temp, efficiency, gamma_pmax, t_noct)
    return normalize_to_8760(cf)


def _irradiance_to_hourly_cf(
    ghi_w: np.ndarray,
    temp_c: np.ndarray,
    efficiency: float,
    gamma_pmax: float,
    t_noct: float,
) -> np.ndarray:
    """Convert hourly GHI and temperature to hourly capacity factors.

    Uses the NOCT cell temperature model with temperature derating.
    """
    ghi = np.asarray(ghi_w, dtype=float)
    temp = np.asarray(temp_c, dtype=float)

    # Cell temperature using NOCT model
    t_cell = temp + (t_noct - 20.0) / 800.0 * ghi

    # Temperature correction factor (gamma_pmax in %/C)
    temp_factor = 1.0 + (gamma_pmax / 100.0) * (t_cell - 25.0)
    temp_factor = np.clip(temp_factor, 0.0, 1.5)

    # CF = GHI/1000 * temp_factor (at STC, 1000 W/m2 gives CF=1)
    cf = (ghi / 1000.0) * temp_factor
    return np.clip(cf, 0.0, 1.0)


def _compute_atlite(
    lat: float,
    lon: float,
    year: int,
    efficiency: float,
    gamma_pmax: float,
    t_noct: float,
    tilt: Optional[float],
    azimuth: float,
    tracking: str,
) -> np.ndarray:
    """Download ERA5 via atlite cutout and compute PV capacity factors."""
    try:
        import atlite
    except ImportError:
        logger.error(
            "atlite is not installed. Install with: pip install solarex[era5]"
        )
        return np.zeros(_HOURS_PER_YEAR)

    import tempfile
    from pathlib import Path

    tmpdir = Path(tempfile.mkdtemp(prefix="solarex_era5_"))
    cutout_path = tmpdir / "cutout.nc"

    delta = 0.05
    cutout = atlite.Cutout(
        path=cutout_path,
        module="era5",
        x=slice(lon - delta, lon + delta),
        y=slice(lat - delta, lat + delta),
        time=str(year),
    )
    cutout.prepare()

    panel_config = {
        "model": "huld",
        "efficiency": efficiency,
        "c_temp_amb": 1.0,
        "c_temp_irrad": 0.035,
        "r_tmod": 298.0,
        "r_tamb": 293.0,
        "r_irradiance": 1000.0,
        "inverter_efficiency": 0.96,
    }

    if tilt is None:
        slope = abs(lat)
        az = 180.0 if lat >= 0 else 0.0
    else:
        slope = tilt
        az = azimuth

    pv_kwargs: dict = {
        "panel": panel_config,
        "orientation": {"slope": slope, "azimuth": az},
        "capacity_factor_timeseries": True,
    }
    if tracking in ("horizontal", "vertical", "dual"):
        pv_kwargs["tracking"] = tracking

    cf_ts = cutout.pv(**pv_kwargs)

    cf_values = cf_ts.values
    if cf_values.ndim == 3:
        cf_hourly = cf_values[:, 0, 0]
    elif cf_values.ndim == 2:
        cf_hourly = cf_values[:, 0]
    else:
        cf_hourly = cf_values

    cf_hourly = np.clip(cf_hourly, 0.0, 1.0)
    return normalize_to_8760(cf_hourly)


def normalize_to_8760(cf: np.ndarray) -> np.ndarray:
    """Ensure output has exactly 8760 hours (remove Feb 29 for leap years)."""
    n = len(cf)
    if n == _HOURS_PER_YEAR:
        return cf
    if n == 8784:
        return np.concatenate([cf[:_FEB29_START], cf[_FEB29_END:]])
    if n > _HOURS_PER_YEAR:
        return cf[:_HOURS_PER_YEAR]
    padded = np.zeros(_HOURS_PER_YEAR)
    padded[:n] = cf
    return padded
