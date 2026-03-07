"""Development zone generation for solar PV sites.

Uses DBSCAN clustering + hull + buffer algorithm.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def generate_development_zones(
    results_gdf,
    min_cf: float,
    min_mcda_score: float,
    buffer_km: float,
    grid_resolution_deg: float = 0.25,
    installation_type: str = "ground",
):
    """Cluster feasible solar PV sites into development zone polygons.

    Parameters
    ----------
    results_gdf : GeoDataFrame
        Per-cell results with ``capacity_factor``, ``mcda_score`` columns.
    min_cf : float
        Minimum capacity factor to include.
    min_mcda_score : float
        Minimum MCDA composite score (0-1) to include.
    buffer_km : float
        Buffer distance around cluster polygons.
    grid_resolution_deg : float
        Grid resolution in degrees for DBSCAN eps.
    installation_type : str
        "ground" or "floating".

    Returns
    -------
    GeoDataFrame
        Zone polygons with zone_id, area_km2, num_sites, avg_cf,
        avg_mcda, total_capacity_mw columns.
    """
    import geopandas as gpd
    from shapely.geometry import MultiPoint
    from sklearn.cluster import DBSCAN

    _EMPTY_COLS = [
        "zone_id", "geometry", "area_km2", "num_sites",
        "avg_cf", "avg_mcda", "total_capacity_mw",
    ]

    feasible = results_gdf[
        (results_gdf["capacity_factor"] >= min_cf)
        & (results_gdf["mcda_score"] >= min_mcda_score)
    ].copy()

    if feasible.empty:
        return gpd.GeoDataFrame(
            columns=_EMPTY_COLS, geometry="geometry", crs="EPSG:4326",
        )

    # Project to metric CRS
    utm_crs = feasible.estimate_utm_crs()
    feasible_utm = feasible.to_crs(utm_crs)

    coords_m = np.column_stack([
        feasible_utm.geometry.x,
        feasible_utm.geometry.y,
    ])

    # 1. Cluster with DBSCAN
    eps_m = grid_resolution_deg * 111_320.0 * 1.5
    clustering = DBSCAN(eps=eps_m, min_samples=1).fit(coords_m)
    feasible_utm = feasible_utm.copy()
    feasible_utm["cluster"] = clustering.labels_

    # 2. Land/water mask
    clip_mask = None
    if installation_type == "floating":
        half_cell_m = grid_resolution_deg * 111_320.0 / 2.0
        all_utm = results_gdf.to_crs(utm_crs)
        clip_mask = all_utm.geometry.buffer(
            half_cell_m, cap_style="square",
        ).union_all()
    else:
        if "lulc_score" in results_gdf.columns:
            land_cells = results_gdf[results_gdf["lulc_score"] > 0]
        else:
            land_cells = results_gdf
        if not land_cells.empty:
            half_cell_m = grid_resolution_deg * 111_320.0 / 2.0
            land_utm = land_cells.to_crs(utm_crs)
            clip_mask = land_utm.geometry.buffer(
                half_cell_m, cap_style="square",
            ).union_all()

    # 3. Build polygon per cluster
    buffer_m = buffer_km * 1000.0
    zones: list[dict] = []
    capacity_density_mw_km2 = 30.0  # ~30 MW/km2 for utility-scale solar

    for cluster_id in sorted(feasible_utm["cluster"].unique()):
        if cluster_id == -1:
            continue

        members = feasible_utm[feasible_utm["cluster"] == cluster_id]
        n = len(members)
        points = MultiPoint(list(members.geometry))

        if n == 1:
            zone_geom = points.buffer(buffer_m)
        elif n == 2:
            zone_geom = points.convex_hull.buffer(buffer_m)
        else:
            try:
                from shapely import concave_hull
                hull = concave_hull(points, ratio=0.3)
            except (ImportError, Exception):
                hull = points.convex_hull
            if hull.is_empty or hull.geom_type in ("Point", "LineString"):
                hull = points.convex_hull
            zone_geom = hull.buffer(buffer_m)

        if zone_geom.is_empty:
            continue

        if clip_mask is not None:
            zone_geom = zone_geom.intersection(clip_mask)
            if zone_geom.is_empty:
                continue

        area_km2 = zone_geom.area / 1e6
        zones.append({
            "zone_id": f"solar_pv_zone_{cluster_id}",
            "geometry": zone_geom,
            "area_km2": area_km2,
            "num_sites": n,
            "avg_cf": float(members["capacity_factor"].mean()),
            "avg_mcda": float(members["mcda_score"].mean()),
            "total_capacity_mw": area_km2 * capacity_density_mw_km2,
        })

    if not zones:
        return gpd.GeoDataFrame(
            columns=_EMPTY_COLS, geometry="geometry", crs="EPSG:4326",
        )

    zones_gdf = gpd.GeoDataFrame(zones, crs=utm_crs).to_crs("EPSG:4326")
    return zones_gdf
