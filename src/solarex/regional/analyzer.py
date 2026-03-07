"""Regional solar PV resource assessment — SolarPVAnalyzer.

No Qt dependency. Uses threading + callbacks for async operation.

Phases:
1. Solar CF computation (0-30%)
2. DEM -> elevation + slope (30-50%)
3. LULC suitability (50-65%)
4. Distance to grid (65-70%)
5. MCDA scoring (70-90%)
6. Zone generation (90-100%)
"""

from __future__ import annotations

import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

import numpy as np

from ..config import SolarConfig, MCDAConfig, FLOATING_LULC_OVERRIDES
from ..results import SolarAnalysisSummary, HourlyIrradianceData

logger = logging.getLogger(__name__)


class SolarPVAnalyzer:
    """Run solar PV resource assessment.

    Parameters
    ----------
    bounds : tuple
        (south, west, north, east) bounding box.
    solar_config : SolarConfig
        Assessment configuration.
    mcda_config : MCDAConfig
        MCDA configuration.
    transmission_lines : list or None
        Transmission line geometries for distance computation.
    """

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        solar_config: SolarConfig | None = None,
        mcda_config: MCDAConfig | None = None,
        transmission_lines: list | None = None,
    ):
        self.south, self.west, self.north, self.east = bounds
        self.solar_config = solar_config or SolarConfig()
        self.mcda_config = mcda_config or MCDAConfig()
        self.transmission_lines = transmission_lines or []
        self._cancelled = False
        self._hourly_data: dict = {}

    def cancel(self):
        """Request cancellation of a running analysis."""
        self._cancelled = True

    def run(
        self,
        on_progress: Optional[Callable[[int, str], None]] = None,
        on_finished: Optional[Callable[[SolarAnalysisSummary], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> SolarAnalysisSummary:
        """Run analysis synchronously. Returns result directly."""
        try:
            result = self._analyze(on_progress)
            if on_finished:
                on_finished(result)
            return result
        except Exception as exc:
            logger.exception("SolarPVAnalyzer error")
            if on_error:
                on_error(str(exc))
            raise

    def run_async(
        self,
        on_progress: Optional[Callable[[int, str], None]] = None,
        on_finished: Optional[Callable[[SolarAnalysisSummary], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> threading.Thread:
        """Run analysis in a background thread. Returns thread handle."""
        t = threading.Thread(
            target=self._run_impl,
            args=(on_progress, on_finished, on_error),
            daemon=True,
        )
        t.start()
        return t

    def _run_impl(self, on_progress, on_finished, on_error):
        try:
            result = self._analyze(on_progress)
            if not self._cancelled and on_finished:
                on_finished(result)
        except Exception as exc:
            logger.exception("SolarPVAnalyzer error")
            if on_error:
                on_error(str(exc))

    def _analyze(
        self, on_progress: Optional[Callable] = None,
    ) -> SolarAnalysisSummary:
        cfg = self.solar_config
        mcda = self.mcda_config

        def _progress(pct, msg):
            if on_progress:
                on_progress(pct, msg)

        # -- Phase 1: Solar capacity factors --
        _progress(2, "Computing solar PV capacity factors...")
        mean_cf, ghi_annual, cf_lats, cf_lons = self._compute_capacity_factors(_progress)
        if self._cancelled:
            return self._empty_summary()

        # Build evaluation grid
        grid_points: list[dict] = []
        for i, lat in enumerate(cf_lats):
            for j, lon in enumerate(cf_lons):
                cf_val = float(mean_cf[i, j]) if mean_cf.size > 0 else 0.0
                ghi_val = float(ghi_annual[i, j]) if ghi_annual is not None and ghi_annual.size > 0 else 0.0
                if not np.isnan(cf_val):
                    grid_points.append({
                        "lat": float(lat),
                        "lon": float(lon),
                        "capacity_factor": cf_val,
                        "ghi_kwh_m2": ghi_val,
                    })

        if not grid_points:
            _progress(100, "No valid grid points in domain")
            return self._empty_summary()

        _progress(30, f"Computed CF for {len(grid_points)} grid points")
        if self._cancelled:
            return self._empty_summary()

        # -- Phase 2: Terrain (elevation + slope) --
        _progress(32, "Fetching terrain data...")
        try:
            from ..data.terrain import fetch_terrain
            elevations, slopes = fetch_terrain(
                grid_points, self.south, self.west, self.north, self.east,
                cfg.grid_resolution,
            )
        except Exception as exc:
            logger.warning("Terrain fetch failed: %s. Using defaults.", exc)
            elevations = [0.0] * len(grid_points)
            slopes = [0.0] * len(grid_points)

        for i, pt in enumerate(grid_points):
            pt["elevation"] = elevations[i]
            pt["slope"] = slopes[i]

        if self._cancelled:
            return self._empty_summary()
        _progress(50, "Terrain data processed")

        # -- Phase 3: LULC suitability --
        _progress(52, "Fetching LULC data...")
        try:
            from ..data.lulc import fetch_lulc
            scores_map = dict(mcda.lulc_scores)
            if cfg.installation == "floating":
                scores_map.update(FLOATING_LULC_OVERRIDES)
            lulc_scores = fetch_lulc(
                grid_points, self.south, self.west, self.north, self.east,
                scores=scores_map,
            )
        except Exception as exc:
            logger.warning("LULC fetch failed: %s. Using defaults.", exc)
            lulc_scores = [0.5] * len(grid_points)

        for i, pt in enumerate(grid_points):
            pt["lulc_score"] = lulc_scores[i]

        if self._cancelled:
            return self._empty_summary()
        _progress(65, "LULC data processed")

        # -- Phase 4: Distance to grid --
        _progress(67, "Computing distance to transmission grid...")
        from ..analysis.grid import compute_dist_to_grid
        for pt in grid_points:
            pt["dist_grid_km"] = compute_dist_to_grid(
                pt["lat"], pt["lon"],
                self.transmission_lines,
                self.south, self.north,
            )

        if self._cancelled:
            return self._empty_summary()
        _progress(70, "Grid distance computed")

        # -- Phase 5: MCDA scoring --
        _progress(72, f"Running MCDA ({mcda.method} weighting)...")
        from ..analysis.mcda import compute_mcda_scores

        criteria_dict = {
            name: {
                "enabled": c.enabled,
                "weight": c.weight,
                "direction": c.direction,
            }
            for name, c in mcda.criteria.items()
        }
        scores, weights = compute_mcda_scores(grid_points, criteria_dict, mcda.method)

        computed_weights = {}
        enabled_names = [n for n, c in mcda.criteria.items() if c.enabled]
        for j, name in enumerate(enabled_names):
            if j < len(weights):
                computed_weights[name] = float(weights[j])

        for i, pt in enumerate(grid_points):
            pt["mcda_score"] = float(scores[i]) if i < len(scores) else 0.0

        _progress(90, "MCDA scoring complete")

        # -- Build summary --
        cf_vals = np.array([pt["capacity_factor"] for pt in grid_points])
        ghi_vals = np.array([pt["ghi_kwh_m2"] for pt in grid_points])
        mcda_vals = np.array([pt["mcda_score"] for pt in grid_points])
        feasible_mask = cf_vals >= cfg.min_capacity_factor

        if feasible_mask.any():
            cell_area_km2 = (
                cfg.grid_resolution * 111.32
                * cfg.grid_resolution * 111.32
                * math.cos(math.radians((self.south + self.north) / 2))
            )
            total_cap = int(feasible_mask.sum()) * cell_area_km2 * 30.0  # MW
        else:
            total_cap = 0.0

        feasible_mcda = mcda_vals[feasible_mask] if feasible_mask.any() else mcda_vals

        _progress(
            100,
            f"Analysis complete: {int(feasible_mask.sum())} feasible cells "
            f"(CF >= {cfg.min_capacity_factor:.0%})",
        )

        return SolarAnalysisSummary(
            total_cells=len(grid_points),
            feasible_cells=int(feasible_mask.sum()),
            cf_min=float(cf_vals.min()),
            cf_max=float(cf_vals.max()),
            cf_avg=float(cf_vals.mean()),
            ghi_avg=float(ghi_vals.mean()) if len(ghi_vals) > 0 else 0.0,
            mcda_score_min=float(feasible_mcda.min()) if len(feasible_mcda) > 0 else 0,
            mcda_score_max=float(feasible_mcda.max()) if len(feasible_mcda) > 0 else 0,
            total_capacity_mw=total_cap,
            computed_weights=computed_weights,
            hourly_data=self._hourly_data if self._hourly_data else None,
        )

    # ------------------------------------------------------------------
    # Phase 1: Solar PV capacity factors
    # ------------------------------------------------------------------

    def _compute_capacity_factors(self, _progress):
        src = self.solar_config.data_source
        if src == "era5_atlite":
            return self._compute_cf_atlite(_progress)
        elif src == "nasa_power":
            return self._compute_cf_nasa_power(_progress)
        else:
            return self._compute_cf_open_meteo(_progress)

    def _compute_cf_atlite(self, _progress):
        from ..data.era5 import compute_cf_atlite as _era5
        cfg = self.solar_config
        return _era5(
            self.south, self.west, self.north, self.east,
            cfg.year, cfg.module_efficiency, cfg.orientation,
            cfg.tilt, cfg.azimuth, cfg.tracking, _progress,
        )

    def _compute_cf_open_meteo(self, _progress):
        from ..data.open_meteo import fetch_open_meteo_solar
        from ..analysis.grid import build_evaluation_grid

        cfg = self.solar_config
        _progress(5, "Fetching solar data from Open-Meteo (ERA5)...")

        lats, lons = build_evaluation_grid(
            self.south, self.west, self.north, self.east, cfg.grid_resolution,
        )
        n_lat, n_lon = len(lats), len(lons)
        mean_cf = np.full((n_lat, n_lon), np.nan)
        ghi_annual = np.full((n_lat, n_lon), np.nan)

        tasks = [(lat, lon) for lat in lats for lon in lons]
        total_pts = len(tasks)

        def _fetch_one(coords):
            la, lo = coords
            result = fetch_open_meteo_solar(la, lo, cfg.year)
            if result is None:
                return la, lo, np.nan, np.nan, None
            ghi_w, temp_c, timestamps = result
            cf = _solar_cf_from_irradiance(
                ghi_w, temp_c, cfg.module_efficiency,
                cfg.module_gamma_pmax, cfg.module_t_noct,
            )
            ghi_ann = float(np.nansum(ghi_w)) / 1000.0
            return la, lo, cf, ghi_ann, (ghi_w, temp_c, timestamps)

        n_workers = min(cfg.effective_workers, total_pts)
        done = 0

        with ThreadPoolExecutor(max_workers=max(1, n_workers)) as pool:
            futures = [pool.submit(_fetch_one, t) for t in tasks]
            for future in as_completed(futures):
                if self._cancelled:
                    pool.shutdown(wait=False, cancel_futures=True)
                    return mean_cf, ghi_annual, lats, lons

                la, lo, cf, ghi_ann, raw = future.result()
                i = int(np.searchsorted(lats, la))
                j = int(np.searchsorted(lons, lo))
                if i < n_lat and j < n_lon:
                    mean_cf[i, j] = cf
                    ghi_annual[i, j] = ghi_ann
                if raw is not None:
                    ghi_w, temp_c, timestamps = raw
                    self._hourly_data[(float(la), float(lo))] = HourlyIrradianceData(
                        timestamps=timestamps, ghi=ghi_w, temperature=temp_c,
                    )

                done += 1
                if done % max(1, total_pts // 20) == 0:
                    pct = 5 + int(25 * done / total_pts)
                    _progress(pct, f"Open-Meteo: {done}/{total_pts} points...")

        _progress(30, f"Open-Meteo: fetched {total_pts} grid points")
        return mean_cf, ghi_annual, lats, lons

    def _compute_cf_nasa_power(self, _progress):
        from ..data.nasa_power import fetch_nasa_power_solar
        from ..analysis.grid import build_evaluation_grid

        cfg = self.solar_config
        _progress(5, "Fetching solar data from NASA POWER (MERRA-2)...")

        lats, lons = build_evaluation_grid(
            self.south, self.west, self.north, self.east, cfg.grid_resolution,
        )
        n_lat, n_lon = len(lats), len(lons)
        mean_cf = np.full((n_lat, n_lon), np.nan)
        ghi_annual = np.full((n_lat, n_lon), np.nan)

        tasks = [(lat, lon) for lat in lats for lon in lons]
        total_pts = len(tasks)

        def _fetch_one(coords):
            la, lo = coords
            result = fetch_nasa_power_solar(la, lo, cfg.year)
            if result is None:
                return la, lo, np.nan, np.nan, None
            ghi_w, temp_c, timestamps = result
            cf = _solar_cf_from_irradiance(
                ghi_w, temp_c, cfg.module_efficiency,
                cfg.module_gamma_pmax, cfg.module_t_noct,
            )
            ghi_ann = float(np.nansum(ghi_w)) / 1000.0
            return la, lo, cf, ghi_ann, (ghi_w, temp_c, timestamps)

        n_workers = min(cfg.effective_workers, total_pts)
        done = 0

        with ThreadPoolExecutor(max_workers=max(1, n_workers)) as pool:
            futures = [pool.submit(_fetch_one, t) for t in tasks]
            for future in as_completed(futures):
                if self._cancelled:
                    pool.shutdown(wait=False, cancel_futures=True)
                    return mean_cf, ghi_annual, lats, lons

                la, lo, cf, ghi_ann, raw = future.result()
                i = int(np.searchsorted(lats, la))
                j = int(np.searchsorted(lons, lo))
                if i < n_lat and j < n_lon:
                    mean_cf[i, j] = cf
                    ghi_annual[i, j] = ghi_ann
                if raw is not None:
                    ghi_w, temp_c, timestamps = raw
                    self._hourly_data[(float(la), float(lo))] = HourlyIrradianceData(
                        timestamps=timestamps, ghi=ghi_w, temperature=temp_c,
                    )

                done += 1
                if done % max(1, total_pts // 20) == 0:
                    pct = 5 + int(25 * done / total_pts)
                    _progress(pct, f"NASA POWER: {done}/{total_pts} points...")

        _progress(30, f"NASA POWER: fetched {total_pts} grid points")
        return mean_cf, ghi_annual, lats, lons

    @staticmethod
    def _empty_summary() -> SolarAnalysisSummary:
        return SolarAnalysisSummary()


def _solar_cf_from_irradiance(
    ghi_w,
    temp_c,
    efficiency: float,
    gamma_pmax: float,
    t_noct: float,
) -> float:
    """Compute mean PV capacity factor from hourly GHI and temperature."""
    ghi = np.asarray(ghi_w, dtype=float)
    temp = np.asarray(temp_c, dtype=float)

    t_cell = temp + (t_noct - 20.0) / 800.0 * ghi
    temp_factor = 1.0 + (gamma_pmax / 100.0) * (t_cell - 25.0)
    temp_factor = np.clip(temp_factor, 0.0, 1.5)

    cf = float(np.nanmean((ghi / 1000.0) * temp_factor))
    return max(0.0, min(cf, 1.0))
