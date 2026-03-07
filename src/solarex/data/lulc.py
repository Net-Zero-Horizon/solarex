"""Land Use / Land Cover data from ESA WorldCover 2021.

Downloads 10m GeoTIFF tiles from S3 and maps LULC classes to suitability scores.
"""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def fetch_lulc(
    grid_points: list[dict],
    south: float,
    west: float,
    north: float,
    east: float,
    scores: dict[int, float] | None = None,
    default_score: float = 0.5,
) -> list[float]:
    """Fetch ESA WorldCover 2021 LULC class and map to suitability score.

    Parameters
    ----------
    grid_points : list[dict]
        Each dict must have "lat" and "lon" keys.
    south, west, north, east : float
        Bounding box.
    scores : dict or None
        Mapping of LULC class code to suitability (0-1).
    default_score : float
        Score for unknown classes.

    Returns
    -------
    list[float]
        Suitability score per grid point.
    """
    if scores is None:
        from ..config import DEFAULT_LULC_SCORES
        scores = dict(DEFAULT_LULC_SCORES)

    try:
        return _fetch_lulc_worldcover(
            grid_points, south, west, north, east, scores, default_score,
        )
    except Exception as exc:
        logger.warning("WorldCover LULC fetch failed: %s", exc)
        return [default_score] * len(grid_points)


def _fetch_lulc_worldcover(
    grid_points: list[dict],
    south: float,
    west: float,
    north: float,
    east: float,
    scores: dict[int, float],
    default_score: float,
) -> list[float]:
    """Download ESA WorldCover 2021 tiles and sample at grid points."""
    import rasterio
    import requests

    tmpdir = Path(tempfile.mkdtemp(prefix="solarex_lulc_"))

    tile_size = 3
    min_lat = int(math.floor(south / tile_size) * tile_size)
    max_lat = int(math.ceil(north / tile_size) * tile_size)
    min_lon = int(math.floor(west / tile_size) * tile_size)
    max_lon = int(math.ceil(east / tile_size) * tile_size)

    lulc_results = [default_score] * len(grid_points)

    for tile_lat in range(min_lat, max_lat, tile_size):
        for tile_lon in range(min_lon, max_lon, tile_size):
            lat_str = f"N{abs(tile_lat):02d}" if tile_lat >= 0 else f"S{abs(tile_lat):02d}"
            lon_str = f"E{abs(tile_lon):03d}" if tile_lon >= 0 else f"W{abs(tile_lon):03d}"
            tile_name = f"ESA_WorldCover_10m_2021_v200_{lat_str}{lon_str}_Map.tif"
            tile_url = (
                "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
                f"v200/2021/map/{tile_name}"
            )
            tile_path = tmpdir / tile_name

            try:
                resp = requests.get(tile_url, timeout=60, stream=True)
                resp.raise_for_status()
                with open(tile_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)

                with rasterio.open(tile_path) as src:
                    for i, pt in enumerate(grid_points):
                        if not (
                            tile_lat <= pt["lat"] < tile_lat + tile_size
                            and tile_lon <= pt["lon"] < tile_lon + tile_size
                        ):
                            continue
                        try:
                            row, col = rasterio.transform.rowcol(
                                src.transform, pt["lon"], pt["lat"],
                            )
                            row = min(max(row, 0), src.height - 1)
                            col = min(max(col, 0), src.width - 1)
                            window = rasterio.windows.Window(col, row, 1, 1)
                            data = src.read(1, window=window)
                            lulc_class = int(data[0, 0])
                            lulc_results[i] = scores.get(lulc_class, default_score)
                        except Exception:
                            pass

            except Exception as exc:
                logger.debug("Could not fetch tile %s: %s", tile_name, exc)

    return lulc_results
