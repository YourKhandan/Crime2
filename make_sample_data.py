"""Creates sample_crime_data.csv: SYNTHETIC offences around a city centre, with a few bad rows.

Run it with:  python make_sample_data.py
The data is made up. 
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
CENTRE_LAT, CENTRE_LON = 41.8781, -87.6298
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = 111_320.0 * np.cos(np.radians(CENTRE_LAT))
OFFENCES = ["THEFT", "BURGLARY", "ASSAULT", "ROBBERY", "VANDALISM", "NARCOTICS", "MOTOR VEHICLE THEFT", "FRAUD"]

# (east m, north m, spread m, incidents, offence weights)
HOTSPOTS = [
    (0, 0, 350, 650, [5, 1, 2, 3, 1, 1, 1, 2]),
    (2600, 1800, 250, 450, [2, 1, 5, 2, 1, 4, 1, 0]),
    (-2800, 900, 450, 500, [2, 4, 1, 1, 3, 1, 3, 1]),
    (1500, -2900, 300, 400, [1, 2, 2, 1, 1, 6, 2, 1]),
    (-1800, -2500, 220, 300, [3, 1, 1, 4, 1, 1, 1, 4]),
    (4200, -800, 200, 250, [1, 1, 3, 1, 5, 1, 1, 1]),
]
rows = []
for east, north, spread, count, weights in HOTSPOTS:
    p = np.array(weights, float) / sum(weights)
    e = rng.normal(east, spread, count)
    n = rng.normal(north, spread, count)
    off = rng.choice(OFFENCES, count, p=p)
    rows += list(zip(CENTRE_LAT + n / M_PER_DEG_LAT, CENTRE_LON + e / M_PER_DEG_LON, off))

# one hotspot stretched along a street (not round), where density-based methods shine
t = rng.uniform(-3000, 3000, 350)
e, n = t + rng.normal(0, 60, 350), 0.45 * t - 4200 + rng.normal(0, 60, 350)
rows += list(zip(CENTRE_LAT + n / M_PER_DEG_LAT, CENTRE_LON + e / M_PER_DEG_LON, rng.choice(OFFENCES, 350)))

# scattered background incidents
e, n = rng.uniform(-6000, 6000, 350), rng.uniform(-6000, 6000, 350)
rows += list(zip(CENTRE_LAT + n / M_PER_DEG_LAT, CENTRE_LON + e / M_PER_DEG_LON, rng.choice(OFFENCES, 350)))

df = pd.DataFrame(rows, columns=["Latitude", "Longitude", "Offence"]).sample(frac=1, random_state=1).reset_index(drop=True)
df["Latitude"], df["Longitude"] = df["Latitude"].round(6), df["Longitude"].round(6)

# extra columns that the app should ignore
df.insert(0, "Case_ID", [f"C{100000 + i}" for i in range(len(df))])
df["Date"] = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, len(df)), unit="D")
df["District"] = rng.integers(1, 25, len(df))
df["Arrest"] = rng.choice([True, False], len(df), p=[0.2, 0.8])

# deliberately broken rows so you can see the coordinate check work
bad = [
    (95.2, -87.6, "THEFT"), (41.9, -200.5, "ASSAULT"), (None, -87.6, "THEFT"), (41.9, None, "BURGLARY"),
    ("unknown", "n/a", "FRAUD"), (0.0, 0.0, "ROBBERY"), (-87.6298, 41.8781, "THEFT"),  # swapped
    (51.5072, -0.1276, "THEFT"),  # a point on another continent
    (41.88, 1e9, "NARCOTICS"), (float("inf"), -87.6, "VANDALISM"),
]
df = df.astype({"Latitude": object, "Longitude": object})
for i, (la, lo, off) in enumerate(bad):
    pos = int(rng.integers(0, len(df)))
    df.loc[pos, ["Latitude", "Longitude", "Offence"]] = [la, lo, off]
df.loc[5, "Offence"] = ""  # a blank offence is kept and labelled "Unknown"

df.to_csv("sample_crime_data.csv", index=False)
print(f"wrote sample_crime_data.csv with {len(df)} rows")
