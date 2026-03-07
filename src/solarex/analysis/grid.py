"""Grid evaluation utilities for solar PV site assessment."""

from __future__ import annotations

import math

import numpy as np


def build_evaluation_grid(
    south: float,
    west: float,
    north: float,
    east: float,
    resolution: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Build regular lat/lon grid within domain bounds.

    Parameters
    ----------
    south, west, north, east : float
        Bounding box (WGS84).
    resolution : float
        Grid spacing in degrees.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (lats, lons) 1D arrays.
    """
    lats = np.arange(south + resolution / 2, north, resolution)
    lons = np.arange(west + resolution / 2, east, resolution)
    if len(lats) == 0:
        lats = np.array([(south + north) / 2])
    if len(lons) == 0:
        lons = np.array([(west + east) / 2])
    return lats, lons


def compute_dist_to_grid(
    lat: float,
    lon: float,
    transmission_lines: list[dict],
    south: float = 0.0,
    north: float = 0.0,
) -> float:
    """Compute minimum distance (km) from point to nearest transmission line.

    Parameters
    ----------
    lat, lon : float
        Point coordinates.
    transmission_lines : list[dict]
        Each dict has a "coords" key with list of [lat, lon] pairs.
    south, north : float
        Domain bounds for mid-latitude correction.

    Returns
    -------
    float
        Distance in km.
    """
    if not transmission_lines:
        return 0.0

    km_per_deg_lat = 111.32
    mid_lat = (south + north) / 2 if (south != 0 or north != 0) else lat
    km_per_deg_lon = 111.32 * math.cos(math.radians(mid_lat))

    min_dist = float("inf")
    for line in transmission_lines:
        coords = line.get("coords", [])
        for coord in coords:
            clat, clon = coord[0], coord[1]
            dy = (lat - clat) * km_per_deg_lat
            dx = (lon - clon) * km_per_deg_lon
            dist = math.sqrt(dx * dx + dy * dy)
            min_dist = min(min_dist, dist)

    return min_dist if min_dist < float("inf") else 0.0
