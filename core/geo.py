"""Small geography helpers."""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


def to_metres(lat, lon) -> np.ndarray:
    """Project latitude/longitude (degrees) onto a flat x/y plane measured in metres.

    Clustering needs real distances. One degree of longitude is shorter than one degree of
    latitude, so clustering raw degrees would squash the map sideways. An equirectangular
    projection around the data's own centre is accurate to a fraction of a percent at city scale
    (it gets rougher for data spread across hundreds of kilometres).
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0 = np.radians(np.median(lat))
    lon0 = np.radians(np.median(lon))
    x = (np.radians(lon) - lon0) * np.cos(lat0) * EARTH_RADIUS_M
    y = (np.radians(lat) - lat0) * EARTH_RADIUS_M
    return np.column_stack([x, y])


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in kilometres (works on numbers or arrays)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * (EARTH_RADIUS_M / 1000.0) * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def extent_km(lat, lon) -> tuple[float, float]:
    """Width and height (km) of the box that holds all points."""
    xy = to_metres(lat, lon)
    width, height = np.ptp(xy, axis=0) / 1000.0
    return float(width), float(height)
