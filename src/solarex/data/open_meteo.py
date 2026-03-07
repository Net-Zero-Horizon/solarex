"""Fetch hourly solar irradiance data from Open-Meteo Historical API (ERA5 reanalysis)."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def fetch_open_meteo_solar(
    lat: float, lon: float, year: int,
) -> "tuple[np.ndarray, np.ndarray, list[str]] | None":
    """Fetch hourly GHI and temperature from Open-Meteo Historical API.

    Returns ``(ghi_w_m2, temp_c, timestamps)`` or None on failure.
    """
    import requests

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": "shortwave_radiation,temperature_2m",
        "timezone": "UTC",
    }

    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        ghi = np.array(hourly.get("shortwave_radiation", []), dtype=float)
        temp = np.array(hourly.get("temperature_2m", []), dtype=float)
        timestamps = hourly.get("time", [])

        if len(ghi) == 0:
            return None

        if len(temp) < len(ghi):
            temp = np.full_like(ghi, 25.0)

        return ghi, temp[: len(ghi)], timestamps[: len(ghi)]

    except Exception as exc:
        logger.debug(
            "Open-Meteo solar fetch failed for (%.2f, %.2f): %s",
            lat, lon, exc,
        )
        return None
