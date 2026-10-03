"""Reading uploaded CSV / Excel files and guessing which columns hold latitude, longitude, offence."""
from __future__ import annotations

import csv
import io
import re

import pandas as pd

# names are compared after lower-casing and removing every non-letter/digit ("Crime_Type" -> "crimetype")
_LAT_EXACT = ["latitude", "lat", "latdeg", "ycoord", "ycoordinate", "y"]
_LON_EXACT = ["longitude", "lon", "lng", "long", "londeg", "xcoord", "xcoordinate", "x"]
_OFFENCE_EXACT = [
    "offence", "offense", "offencetype", "offensetype", "crime", "crimetype", "primarytype",
    "category", "type", "incidenttype", "crimedescription", "description",
]
_LAT_CONTAINS = ["latitude"]
_LON_CONTAINS = ["longitude"]
_OFFENCE_CONTAINS = ["offence", "offense", "crime"]


def _norm(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _words(name) -> set[str]:
    """Words inside a column name: 'crime_lat' and 'CrimeLat' both give {'crime', 'lat'}."""
    return {w.lower() for w in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", str(name))}


def _pick(columns, exact, contains, taken, words=()):
    normed = {c: _norm(c) for c in columns if c not in taken}
    for wanted in exact:
        for col, n in normed.items():
            if n == wanted:
                return col
    for token in contains:
        for col, n in normed.items():
            if token in n:
                return col
    for col in normed:  # last resort: the name contains the word, e.g. crime_lat
        if _words(col) & set(words):
            return col
    return None


def guess_columns(columns) -> dict[str, str | None]:
    """Best guess for the latitude, longitude and offence columns."""
    taken: set = set()
    lat = _pick(columns, _LAT_EXACT, _LAT_CONTAINS, taken, words=("lat",))
    taken.add(lat)
    lon = _pick(columns, _LON_EXACT, _LON_CONTAINS, taken, words=("lon", "lng", "long"))
    taken.add(lon)
    offence = _pick(columns, _OFFENCE_EXACT, _OFFENCE_CONTAINS, taken)
    return {"lat": lat, "lon": lon, "offence": offence}


def _is_excel(filename: str) -> bool:
    return filename.lower().endswith((".xlsx", ".xlsm", ".xls"))


def _detect_separator(raw: bytes) -> str:
    sample = raw[:20_000].decode("utf-8", errors="ignore")
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def read_header(raw: bytes, filename: str) -> list[str]:
    """Column names only (fast, even for big files)."""
    if _is_excel(filename):
        head = pd.read_excel(io.BytesIO(raw), nrows=0)
    else:
        head = pd.read_csv(
            io.BytesIO(raw), nrows=0, sep=_detect_separator(raw), encoding="utf-8-sig", encoding_errors="replace"
        )
    return [str(c) for c in head.columns]


def read_columns(raw: bytes, filename: str, columns: list[str]) -> pd.DataFrame:
    """Read ONLY the requested columns; everything else in the file is skipped."""
    columns = list(dict.fromkeys(columns))  # unique, keep order
    if _is_excel(filename):
        return pd.read_excel(io.BytesIO(raw), usecols=columns)
    return pd.read_csv(
        io.BytesIO(raw),
        usecols=columns,
        sep=_detect_separator(raw),
        encoding="utf-8-sig",
        encoding_errors="replace",
        low_memory=False,
    )
