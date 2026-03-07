"""Solar irradiance characterization — pure functions.

Provides peak sun hours, clearness index, diurnal/monthly profiles,
and performance ratio computation.
"""

from __future__ import annotations

import math

import numpy as np


def compute_peak_sun_hours(ghi_hourly: np.ndarray) -> float:
    """Compute peak sun hours (PSH) from hourly GHI.

    PSH = total irradiation (Wh/m2) / 1000 W/m2 expressed as hours/day.
    """
    ghi = np.asarray(ghi_hourly, dtype=float)
    total_wh = float(np.nansum(ghi))  # each element is W/m2 x 1h
    n_days = max(len(ghi) / 24.0, 1.0)
    return total_wh / 1000.0 / n_days


def compute_performance_ratio(
    ghi_hourly: np.ndarray,
    temp_hourly: np.ndarray,
    efficiency: float,
    gamma_pmax: float,
    t_noct: float,
) -> float:
    """Compute performance ratio (PR) = actual yield / reference yield.

    Reference yield = GHI / G_STC (1000 W/m2).
    Actual yield includes temperature derating via NOCT model.
    """
    ghi = np.asarray(ghi_hourly, dtype=float)
    temp = np.asarray(temp_hourly, dtype=float)

    mask = ghi > 0
    if not np.any(mask):
        return 0.0

    ghi_pos = ghi[mask]
    temp_pos = temp[mask] if len(temp) == len(ghi) else np.full(mask.sum(), 25.0)

    # Cell temperature (NOCT model)
    t_cell = temp_pos + (t_noct - 20.0) / 800.0 * ghi_pos

    # Temperature derating
    temp_factor = 1.0 + (gamma_pmax / 100.0) * (t_cell - 25.0)
    temp_factor = np.clip(temp_factor, 0.0, 1.5)

    # PR = mean(temp_factor) for hours with sunlight
    return float(np.mean(temp_factor))


def compute_clearness_index(
    ghi_hourly: np.ndarray,
    latitude: float,
    timestamps: list[str],
) -> float:
    """Compute clearness index Kt = GHI / extraterrestrial irradiance.

    Uses daily totals averaged over the year.
    """
    from datetime import datetime

    ghi = np.asarray(ghi_hourly, dtype=float)
    if len(ghi) == 0 or len(timestamps) == 0:
        return 0.0

    G_sc = 1361.0  # Solar constant W/m2
    lat_rad = math.radians(latitude)

    daily_ghi: dict[tuple, float] = {}
    daily_et: dict[tuple, float] = {}

    for i, ts in enumerate(timestamps):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue

        doy = dt.timetuple().tm_yday
        hour = dt.hour + dt.minute / 60.0
        day_key = (dt.year, doy)

        # Declination (Cooper's equation)
        decl = 23.45 * math.sin(math.radians(360 * (284 + doy) / 365))
        decl_rad = math.radians(decl)

        # Hour angle
        omega = math.radians(15.0 * (hour - 12.0))

        # Solar zenith angle
        cos_z = (
            math.sin(lat_rad) * math.sin(decl_rad)
            + math.cos(lat_rad) * math.cos(decl_rad) * math.cos(omega)
        )

        # Eccentricity correction
        E0 = 1.0 + 0.033 * math.cos(math.radians(360 * doy / 365))

        et_irrad = max(0.0, G_sc * E0 * cos_z)

        daily_ghi.setdefault(day_key, 0.0)
        daily_et.setdefault(day_key, 0.0)
        if i < len(ghi):
            daily_ghi[day_key] += max(0.0, float(ghi[i]))
        daily_et[day_key] += et_irrad

    total_ghi = sum(daily_ghi.values())
    total_et = sum(daily_et.values())

    if total_et <= 0:
        return 0.0
    return min(1.0, total_ghi / total_et)


def compute_diurnal_irradiance(
    ghi_hourly: np.ndarray,
    timestamps: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean hourly GHI pattern over 24 hours.

    Returns (hours[0..23], mean_ghi[24]) in W/m2.
    """
    from datetime import datetime

    ghi = np.asarray(ghi_hourly, dtype=float)
    hours = np.arange(24)
    sums = np.zeros(24)
    counts = np.zeros(24)

    for i, ts in enumerate(timestamps):
        if i >= len(ghi):
            break
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        h = dt.hour
        sums[h] += ghi[i]
        counts[h] += 1

    means = np.where(counts > 0, sums / counts, 0.0)
    return hours, means


def compute_monthly_irradiance(
    ghi_hourly: np.ndarray,
    timestamps: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute monthly total irradiance.

    Returns (months[1..12], totals[12]) in kWh/m2/month.
    """
    from datetime import datetime

    ghi = np.asarray(ghi_hourly, dtype=float)
    months = np.arange(1, 13)
    totals = np.zeros(12)

    for i, ts in enumerate(timestamps):
        if i >= len(ghi):
            break
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        m = dt.month - 1  # 0-indexed
        totals[m] += max(0.0, ghi[i]) / 1000.0  # W*h -> kWh

    return months, totals
