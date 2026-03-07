"""Data fetching modules for solar resource assessment."""

from .open_meteo import fetch_open_meteo_solar
from .nasa_power import fetch_nasa_power_solar

__all__ = [
    "fetch_open_meteo_solar",
    "fetch_nasa_power_solar",
]
