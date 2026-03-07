"""LCOE sensitivity analysis for solar PV projects."""

from __future__ import annotations

import os
from dataclasses import asdict

from .financial import SolarFinancialInputs, compute_pv_financials


def compute_pv_lcoe_sensitivity(
    inputs: SolarFinancialInputs,
    param_name: str,
    values: list[float],
    max_workers: int = 0,
) -> list[float]:
    """Sweep a parameter and compute LCOE for each value.

    Parameters
    ----------
    inputs : SolarFinancialInputs
        Base financial inputs.
    param_name : str
        Name of the field to vary (e.g. "capex_per_kw", "capacity_factor").
    values : list[float]
        Values to sweep.
    max_workers : int
        Number of threads. 0 = auto.

    Returns
    -------
    list[float]
        LCOE values, one per input value.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _eval(val):
        d = asdict(inputs)
        d[param_name] = val
        modified = SolarFinancialInputs(**d)
        return compute_pv_financials(modified).lcoe

    workers = max_workers if max_workers > 0 else (os.cpu_count() or 4)
    n_workers = min(workers, len(values))

    if n_workers <= 1 or len(values) <= 3:
        return [_eval(v) for v in values]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(_eval, values))
