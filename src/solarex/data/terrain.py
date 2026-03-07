"""Terrain data fetching (DEM elevation + slope).

Fallback chain: SRTM/rasterio -> Open-Meteo Elevation -> Open-Elevation API.
"""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def fetch_terrain(
    grid_points: list[dict],
    south: float,
    west: float,
    north: float,
    east: float,
    grid_resolution: float = 0.25,
) -> tuple[list[float], list[float]]:
    """Fetch elevation and compute slope for grid points.

    Tries SRTM/rasterio first, then Open-Meteo Elevation, then Open-Elevation API.

    Returns (elevations, slopes) lists.
    """
    try:
        return _fetch_terrain_rasterio(grid_points, south, west, north, east)
    except Exception as exc:
        logger.warning("Rasterio terrain failed (%s), trying Open-Meteo...", exc)

    try:
        return _fetch_terrain_open_meteo(grid_points, grid_resolution)
    except Exception as exc:
        logger.warning("Open-Meteo elevation failed (%s), trying Open-Elevation...", exc)

    return _fetch_terrain_api(grid_points, grid_resolution)


def _fetch_terrain_rasterio(
    grid_points: list[dict],
    south: float,
    west: float,
    north: float,
    east: float,
) -> tuple[list[float], list[float]]:
    """Fetch terrain using rasterio with SRTM tiles."""
    import rasterio

    tmpdir = Path(tempfile.mkdtemp(prefix="solarex_dem_"))
    dem_path = tmpdir / "dem.tif"

    try:
        import elevation as elev
        elev.clip(
            bounds=(west, south, east, north),
            output=str(dem_path),
        )
    except (ImportError, Exception):
        raise RuntimeError("elevation package not available")

    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        transform = src.transform

        elevations = []
        for pt in grid_points:
            row, col = rasterio.transform.rowcol(
                transform, pt["lon"], pt["lat"],
            )
            row = min(max(row, 0), dem_data.shape[0] - 1)
            col = min(max(col, 0), dem_data.shape[1] - 1)
            elevations.append(float(dem_data[row, col]))

        # Compute slope from DEM
        res_x = abs(transform.a)
        mid_lat = (south + north) / 2
        cell_size_m = res_x * 111320 * math.cos(math.radians(mid_lat))

        dy, dx = np.gradient(dem_data.astype(float), cell_size_m)
        slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
        slope_deg = np.degrees(slope_rad)

        slopes = []
        for pt in grid_points:
            row, col = rasterio.transform.rowcol(
                transform, pt["lon"], pt["lat"],
            )
            row = min(max(row, 0), slope_deg.shape[0] - 1)
            col = min(max(col, 0), slope_deg.shape[1] - 1)
            slopes.append(float(slope_deg[row, col]))

    return elevations, slopes


def _fetch_terrain_open_meteo(
    grid_points: list[dict],
    grid_resolution: float,
) -> tuple[list[float], list[float]]:
    """Fetch elevation from Open-Meteo Elevation API."""
    import requests

    elevations: list[float] = []
    batch_size = 100

    for start in range(0, len(grid_points), batch_size):
        batch = grid_points[start:start + batch_size]
        lats = ",".join(f"{pt['lat']:.4f}" for pt in batch)
        lons = ",".join(f"{pt['lon']:.4f}" for pt in batch)

        resp = requests.get(
            "https://api.open-meteo.com/v1/elevation",
            params={"latitude": lats, "longitude": lons},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        elev_list = data.get("elevation", [])
        if isinstance(elev_list, (int, float)):
            elev_list = [elev_list]
        for e in elev_list:
            elevations.append(float(e) if e is not None else 0.0)

    while len(elevations) < len(grid_points):
        elevations.append(0.0)

    slopes = _slope_from_elevations(grid_points, elevations, grid_resolution)
    return elevations, slopes


def _fetch_terrain_api(
    grid_points: list[dict],
    grid_resolution: float,
) -> tuple[list[float], list[float]]:
    """Fetch elevation from Open-Elevation API as last-resort fallback."""
    import requests

    elevations = []
    batch_size = 256

    for start in range(0, len(grid_points), batch_size):
        batch = grid_points[start:start + batch_size]
        locations = "|".join(
            f"{pt['lat']},{pt['lon']}" for pt in batch
        )
        try:
            resp = requests.get(
                "https://api.open-elevation.com/api/v1/lookup",
                params={"locations": locations},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            for r in data.get("results", []):
                elevations.append(float(r.get("elevation", 0)))
        except Exception:
            elevations.extend([0.0] * len(batch))

    slopes = _slope_from_elevations(grid_points, elevations, grid_resolution)
    return elevations, slopes


def _slope_from_elevations(
    grid_points: list[dict],
    elevations: list[float],
    grid_resolution: float,
) -> list[float]:
    """Estimate slope from elevation differences between grid neighbours."""
    slopes: list[float] = []
    if len(grid_points) <= 1:
        return [0.0] * len(grid_points)

    for i, pt in enumerate(grid_points):
        neighbours = []
        for j, other in enumerate(grid_points):
            if i == j:
                continue
            dist_deg = math.sqrt(
                (pt["lat"] - other["lat"]) ** 2
                + (pt["lon"] - other["lon"]) ** 2,
            )
            if dist_deg < grid_resolution * 1.5:
                neighbours.append(j)

        if neighbours:
            elev_diffs = [abs(elevations[j] - elevations[i]) for j in neighbours]
            dist_m = grid_resolution * 111320 * math.cos(math.radians(pt["lat"]))
            max_slope = max(elev_diffs) / max(dist_m, 1)
            slopes.append(math.degrees(math.atan(max_slope)))
        else:
            slopes.append(0.0)
    return slopes
