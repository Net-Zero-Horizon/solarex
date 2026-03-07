"""Cell temperature and derating factor computation."""

from __future__ import annotations

import numpy as np


def compute_temp_analysis(
    ghi_hourly: np.ndarray,
    temp_ambient: np.ndarray,
    t_noct: float,
    gamma_pmax: float = -0.40,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cell temperature and derating factor arrays.

    Parameters
    ----------
    ghi_hourly : np.ndarray
        Hourly global horizontal irradiance (W/m2).
    temp_ambient : np.ndarray
        Hourly ambient temperature (C).
    t_noct : float
        Nominal operating cell temperature (C).
    gamma_pmax : float
        Temperature coefficient of power (%/C), typically negative.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (cell_temp_c, derating_factor) arrays.
    """
    ghi = np.asarray(ghi_hourly, dtype=float)
    temp = np.asarray(temp_ambient, dtype=float)

    t_cell = temp + (t_noct - 20.0) / 800.0 * ghi
    derating = 1.0 + (gamma_pmax / 100.0) * (t_cell - 25.0)
    derating = np.clip(derating, 0.0, 1.5)

    return t_cell, derating
