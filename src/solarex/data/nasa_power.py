"""Fetch hourly solar irradiance data from NASA POWER API (MERRA-2 reanalysis)."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def fetch_nasa_power_solar(
    lat: float, lon: float, year: int,
) -> "tuple[np.ndarray, np.ndarray, list[str]] | None":
    """Fetch hourly GHI and temperature from NASA POWER API.

    Returns ``(ghi_w_m2, temp_c, timestamps)`` or None on failure.
    """
    import requests

    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,T2M",
        "community": "RE",
        "longitude": round(lon, 4),
        "latitude": round(lat, 4),
        "start": f"{year}0101",
        "end": f"{year}1231",
        "format": "JSON",
    }

    try:
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        props = data.get("properties", {}).get("parameter", {})
        ghi_dict = props.get("ALLSKY_SFC_SW_DWN", {})
        t2m_dict = props.get("T2M", {})

        timestamps = []
        valid_ghi = []
        valid_temp = []
        for key, val in ghi_dict.items():
            if val == -999:
                continue
            valid_ghi.append(val)
            t_val = t2m_dict.get(key, -999)
            valid_temp.append(t_val if t_val != -999 else 25.0)
            # Key format: "2022010100" (YYYYMMDDHH)
            try:
                ts = f"{key[:4]}-{key[4:6]}-{key[6:8]}T{key[8:10]}:00"
                timestamps.append(ts)
            except (IndexError, ValueError):
                timestamps.append("")

        ghi = np.array(valid_ghi, dtype=float)
        temp = np.array(valid_temp, dtype=float)

        if len(ghi) == 0:
            return None

        return ghi, temp[: len(ghi)], timestamps[: len(ghi)]

    except Exception as exc:
        logger.debug(
            "NASA POWER solar fetch failed for (%.2f, %.2f): %s",
            lat, lon, exc,
        )
        return None
