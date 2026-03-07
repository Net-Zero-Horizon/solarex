"""Configuration dataclasses for SolareX."""

from __future__ import annotations

from dataclasses import dataclass, field


# ESA WorldCover 2021 class codes -> default suitability scores for solar PV
DEFAULT_LULC_SCORES: dict[int, float] = {
    10: 0.1,   # Tree cover (shading)
    20: 0.5,   # Shrubland
    30: 0.9,   # Grassland (good for ground-mount)
    40: 0.6,   # Cropland (agrivoltaics possible)
    50: 0.0,   # Built-up
    60: 1.0,   # Bare / sparse vegetation (ideal)
    70: 0.0,   # Snow and ice
    80: 0.0,   # Permanent water bodies (ground default)
    90: 0.2,   # Herbaceous wetland
    95: 0.0,   # Mangroves
    100: 0.1,  # Moss and lichen
}

# Floating solar overrides
FLOATING_LULC_OVERRIDES: dict[int, float] = {
    80: 0.8,   # Water -> suitable for floating PV
    10: 0.0,   # Tree cover -> exclude
    50: 0.0,   # Built-up -> exclude
}


@dataclass(frozen=True)
class ModuleSpec:
    """Technical specification of a PV module from the CEC/NREL SAM database."""

    key: str                # Full CEC key, e.g. "Canadian_Solar_Inc__CS6U_330P"
    manufacturer: str       # Extracted from key
    name: str               # Display name
    technology: str         # Mono-c-Si, Multi-c-Si, CdTe, CIGS, etc.
    stc_power_w: float      # STC rated power (W)
    ptc_power_w: float      # PTC power (W)
    area_m2: float          # Module area (m^2)
    efficiency: float       # STC/area/1000 (0-1)
    bifacial: bool
    v_oc: float             # Open-circuit voltage (V)
    i_sc: float             # Short-circuit current (A)
    v_mp: float             # Voltage at max power (V)
    i_mp: float             # Current at max power (A)
    gamma_pmax: float       # Temperature coefficient of power (%/C)
    t_noct: float           # NOCT (C)
    n_cells: int            # Number of cells in series
    length_m: float
    width_m: float


@dataclass(frozen=True)
class CriterionConfig:
    """Configuration for a single MCDA criterion."""

    enabled: bool = True
    weight: float = 0.2
    direction: str = "maximize"  # "maximize" or "minimize"


@dataclass
class MCDAConfig:
    """Multi-criteria decision analysis configuration."""

    method: str = "manual"  # "manual" | "entropy" | "pca"
    criteria: dict[str, CriterionConfig] = field(default_factory=lambda: {
        "capacity_factor": CriterionConfig(True, 0.40, "maximize"),
        "slope": CriterionConfig(True, 0.20, "minimize"),
        "elevation": CriterionConfig(True, 0.05, "minimize"),
        "lulc_score": CriterionConfig(True, 0.20, "maximize"),
        "dist_grid_km": CriterionConfig(True, 0.15, "minimize"),
    })
    lulc_scores: dict[int, float] = field(
        default_factory=lambda: dict(DEFAULT_LULC_SCORES),
    )


@dataclass
class SolarConfig:
    """User-configurable solar PV assessment parameters."""

    module_key: str = ""
    module_efficiency: float = 0.20
    module_gamma_pmax: float = -0.40  # %/C temperature coefficient
    module_stc_w: float = 400.0
    module_t_noct: float = 45.0       # NOCT (C)
    orientation: str = "latitude_optimal"  # "latitude_optimal" | "custom"
    tilt: float = 0.0                 # degrees (used when orientation=custom)
    azimuth: float = 180.0            # degrees south-facing
    tracking: str = "none"            # "none" | "horizontal" | "vertical" | "dual"
    installation: str = "ground"      # "ground" | "floating"
    year: int = 2022
    grid_resolution: float = 0.25     # degrees
    min_capacity_factor: float = 0.15
    zone_buffer_km: float = 5.0
    module_capacity_kw: float = 0.4   # kW per module for capacity estimation
    data_source: str = "open_meteo"   # "open_meteo" | "nasa_power" | "era5_atlite"
    parallel_workers: int = 0         # 0 = auto (cpu_count)

    @property
    def effective_workers(self) -> int:
        import os
        if self.parallel_workers > 0:
            return self.parallel_workers
        return os.cpu_count() or 4
