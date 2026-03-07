"""Result dataclasses for SolareX."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HourlyIrradianceData:
    """Hourly solar irradiance and temperature for one grid cell."""

    timestamps: list[str]     # ISO-8601 hourly timestamps
    ghi: Any                  # np.ndarray W/m2
    temperature: Any          # np.ndarray C


@dataclass
class SolarAnalysisSummary:
    """Aggregated solar PV assessment results."""

    total_cells: int = 0
    feasible_cells: int = 0
    cf_min: float = 0.0
    cf_max: float = 0.0
    cf_avg: float = 0.0
    ghi_avg: float = 0.0           # kWh/m2/yr
    mcda_score_min: float = 0.0
    mcda_score_max: float = 0.0
    total_capacity_mw: float = 0.0
    computed_weights: dict[str, float] = field(default_factory=dict)
    results_gdf: Any = None        # GeoDataFrame
    hourly_data: Any = None        # dict[(lat,lon)] -> HourlyIrradianceData

    def compute_statistics(self) -> dict[str, float]:
        """Return summary statistics as a flat dict."""
        return {
            "total_cells": self.total_cells,
            "feasible_cells": self.feasible_cells,
            "cf_min": self.cf_min,
            "cf_max": self.cf_max,
            "cf_avg": self.cf_avg,
            "ghi_avg_kwh_m2": self.ghi_avg,
            "mcda_score_min": self.mcda_score_min,
            "mcda_score_max": self.mcda_score_max,
            "total_capacity_mw": self.total_capacity_mw,
        }

    def to_dict(self) -> dict:
        """Serialize to dictionary (excludes GeoDataFrame/arrays)."""
        d = self.compute_statistics()
        d["computed_weights"] = dict(self.computed_weights)
        return d
