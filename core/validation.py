"""Checks that every latitude and longitude is usable.

Only three columns are ever looked at: latitude, longitude and offence. Every other column in the
file is ignored.

A row is invalid when:
  * latitude or longitude is missing, not a number, or infinite
  * latitude is outside -90..90, or longitude is outside -180..180
  * (optional) both are exactly 0, which is the usual "no location" placeholder
  * (optional) the point is far away from the rest of the data
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geo import haversine_km

PROBLEM_MISSING = "Latitude or longitude is missing, not a number, or infinite"
PROBLEM_LAT_RANGE = "Latitude is outside -90 to 90"
PROBLEM_LON_RANGE = "Longitude is outside -180 to 180"
PROBLEM_SWAPPED = "Latitude and longitude may be swapped"
PROBLEM_ZERO = "Coordinates are exactly 0, 0 (usually a placeholder)"
PROBLEM_FAR = "Point is far from the rest of the data"


@dataclass
class ValidationReport:
    n_rows: int
    n_valid: int
    n_invalid: int
    problem_counts: dict[str, int]  # a row can have more than one problem
    clean: pd.DataFrame  # columns: lat, lon, offence, file_row
    invalid: pd.DataFrame  # columns: file_row, latitude, longitude, offence, problem


def validate_coordinates(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    offence_col: str | None = None,
    *,
    flag_zero_zero: bool = True,
    max_km_from_centre: float | None = None,
) -> ValidationReport:
    n = len(df)
    file_row = np.arange(n) + 2  # row 1 of the file is the header

    lat_raw = df[lat_col].reset_index(drop=True)
    lon_raw = df[lon_col].reset_index(drop=True)
    lat = pd.to_numeric(lat_raw, errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(lon_raw, errors="coerce").to_numpy(dtype=float)

    finite = np.isfinite(lat) & np.isfinite(lon)
    with np.errstate(invalid="ignore"):
        lat_bad = finite & (np.abs(lat) > 90)
        lon_bad = finite & (np.abs(lon) > 180)
        swapped = lat_bad & (np.abs(lon) <= 90)
        zero = finite & (lat == 0) & (lon == 0) if flag_zero_zero else np.zeros(n, dtype=bool)

    far = np.zeros(n, dtype=bool)
    if max_km_from_centre:
        usable = finite & ~lat_bad & ~lon_bad & ~zero
        if usable.sum() >= 3:
            centre_lat, centre_lon = np.median(lat[usable]), np.median(lon[usable])
            km = haversine_km(lat[usable], lon[usable], centre_lat, centre_lon)
            far[np.flatnonzero(usable)[km > float(max_km_from_centre)]] = True

    checks = [
        (PROBLEM_MISSING, ~finite),
        (PROBLEM_LAT_RANGE, lat_bad),
        (PROBLEM_LON_RANGE, lon_bad),
        (PROBLEM_SWAPPED, swapped),
        (PROBLEM_ZERO, zero),
        (PROBLEM_FAR, far),
    ]
    invalid = np.zeros(n, dtype=bool)
    problem = np.full(n, "", dtype=object)
    for text, mask in checks:
        invalid |= mask
        problem[mask] = np.where(problem[mask] == "", text, problem[mask] + "; " + text)

    # offence is kept as plain text; blanks become "Unknown"
    if offence_col is None:
        offence = pd.Series(["Unknown"] * n, dtype=object)
    else:
        offence = df[offence_col].reset_index(drop=True).astype("string").str.strip()
        offence = offence.str.replace(r"\s+", " ", regex=True).fillna("Unknown")
        offence = offence.mask(offence == "", "Unknown").astype(object)

    ok = ~invalid
    clean = pd.DataFrame(
        {"lat": lat[ok], "lon": lon[ok], "offence": offence[ok].to_numpy(), "file_row": file_row[ok]}
    )
    bad = pd.DataFrame(
        {
            "file_row": file_row[invalid],
            "latitude": lat_raw[invalid].to_numpy(),
            "longitude": lon_raw[invalid].to_numpy(),
            "offence": offence[invalid].to_numpy(),
            "problem": problem[invalid],
        }
    )
    return ValidationReport(
        n_rows=n,
        n_valid=int(ok.sum()),
        n_invalid=int(invalid.sum()),
        problem_counts={text: int(mask.sum()) for text, mask in checks if mask.any()},
        clean=clean,
        invalid=bad,
    )
