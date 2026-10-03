"""Tab 1: load a file, check every latitude/longitude, choose what to cluster."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from core.data_io import guess_columns, read_columns, read_header
from core.dataset import Dataset
from core.geo import extent_km, to_metres
from core.validation import ValidationReport, validate_coordinates

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "sample_crime_data.csv"
NO_COLUMN = "(none)"
DROP = "Drop them and continue"
STOP = "Stop so I can fix the file"


@st.cache_data(show_spinner=False)
def _header(raw: bytes, name: str) -> list[str]:
    return read_header(raw, name)


@st.cache_data(show_spinner="Reading and checking the file...")
def _load_and_validate(raw: bytes, name: str, lat_col: str, lon_col: str, offence_col: str | None, flag_zero: bool, far_km: float | None) -> ValidationReport:
    columns = [lat_col, lon_col] + ([offence_col] if offence_col else [])
    df = read_columns(raw, name, columns)  # only these columns are read
    return validate_coordinates(df, lat_col, lon_col, offence_col, flag_zero_zero=flag_zero, max_km_from_centre=far_km)


def _short_hash(*parts) -> str:
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:10]


def render() -> Dataset | None:
    st.subheader("Load your data")
    left, right = st.columns([3, 2], gap="large")
    with left:
        upload = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"], key="upload")
    with right:
        st.write(
            "Each row needs a latitude, a longitude and an offence. Every other column in your file is ignored."
        )
        use_sample = st.checkbox("Use the built-in sample data instead (synthetic, with a few bad rows)", key="use_sample")

    if upload is not None:
        raw, name = upload.getvalue(), upload.name
    elif use_sample and SAMPLE_PATH.exists():
        raw, name = SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name
    else:
        st.info("Upload a file above, or tick the sample data option, to begin.")
        return None

    try:
        columns = _header(raw, name)
    except Exception as exc:  # unreadable or empty file
        st.error(f"Could not read {name}: {exc}. Check that it is a CSV or .xlsx file with a header row.")
        return None
    if len(columns) < 2:
        st.error("The file has fewer than two columns. It needs at least latitude and longitude.")
        return None

    guess = guess_columns(columns)

    st.markdown("#### Choose the columns")
    c1, c2, c3 = st.columns(3)
    lat_col = c1.selectbox("Latitude column", columns, index=columns.index(guess["lat"]) if guess["lat"] else 0, key=f"lat_{name}")
    lon_col = c2.selectbox("Longitude column", columns, index=columns.index(guess["lon"]) if guess["lon"] else min(1, len(columns) - 1), key=f"lon_{name}")
    offence_options = [NO_COLUMN] + columns
    offence_choice = c3.selectbox(
        "Offence column", offence_options, index=offence_options.index(guess["offence"]) if guess["offence"] else 0, key=f"offence_{name}"
    )
    offence_col = None if offence_choice == NO_COLUMN else offence_choice
    if len({lat_col, lon_col, offence_col or "-"}) < (3 if offence_col else 2):
        st.error("Pick a different column for each of latitude, longitude and offence.")
        return None

    st.markdown("#### Coordinate checks")
    with st.container(border=True):
        st.write(
            "Always checked: latitude and longitude must be numbers, latitude must be between -90 and 90, "
            "and longitude must be between -180 and 180."
        )
        a, b = st.columns(2)
        flag_zero = a.checkbox("Treat exactly 0, 0 as invalid", True, key="flag_zero", help="0, 0 is usually what software writes when the real location is unknown.")
        policy = a.radio("If invalid coordinates are found", [DROP, STOP], key="policy")
        use_far = b.checkbox("Also flag points far from the rest of the data", False, key="use_far", help="Catches points that are valid on Earth but clearly in the wrong place, such as another country.")
        far_km = b.number_input("Flag points farther than this from the centre of the data (km)", 1.0, 20000.0, 100.0, step=10.0, key="far_km", disabled=not use_far)

    try:
        report = _load_and_validate(raw, name, lat_col, lon_col, offence_col, flag_zero, float(far_km) if use_far else None)
    except Exception as exc:
        st.error(f"Could not read those columns: {exc}")
        return None

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows in file", f"{report.n_rows:,}", border=True)
    m2.metric("Valid rows", f"{report.n_valid:,}", border=True)
    m3.metric("Invalid rows", f"{report.n_invalid:,}", border=True)
    m4.metric("Offence types", f"{report.clean['offence'].nunique():,}", border=True)

    if report.n_invalid == 0:
        st.success(f"All {report.n_rows:,} latitude and longitude pairs are valid.")
    else:
        message = f"{report.n_invalid:,} of {report.n_rows:,} rows have invalid coordinates."
        if policy == STOP:
            st.error(message + " Fix them in your file and upload it again, or switch the option above to drop them.")
        else:
            st.warning(message + " They are left out of the analysis.")
        with st.expander("See the invalid rows", expanded=policy == STOP):
            st.write("A row can have more than one problem, so the counts below can add up to more than the number of rows.")
            st.dataframe(pd.DataFrame({"Problem": list(report.problem_counts), "Rows": list(report.problem_counts.values())}), hide_index=True)
            st.dataframe(report.invalid.head(500).rename(columns={"file_row": "Row in file"}), hide_index=True)
            st.caption("Row numbers count the header as row 1. The table shows the first 500 invalid rows; the download has all of them.")
            st.download_button("Download invalid rows (CSV)", report.invalid.to_csv(index=False).encode("utf-8"), file_name="invalid_rows.csv", mime="text/csv", key="dl_invalid")
        if policy == STOP:
            return None

    if report.n_valid < 10:
        st.error("Fewer than 10 valid rows are left, which is too few to cluster.")
        return None

    st.markdown("#### Choose what to cluster")
    counts = report.clean["offence"].value_counts()
    f1, f2 = st.columns([3, 1])
    chosen = f1.multiselect(
        "Offence types (leave empty to use all)",
        list(counts.index),
        format_func=lambda o: f"{o} ({counts[o]:,})",
        key=f"offence_filter_{name}_{offence_col}",
    )
    cap = int(f2.number_input("Maximum points to cluster", 1000, 500_000, 30_000, step=1000, key="max_points", help="Bigger data is sampled at random to keep the app responsive."))

    df = report.clean[report.clean["offence"].isin(chosen)] if chosen else report.clean
    if df.empty:
        st.error("No rows match that offence selection.")
        return None
    n_before_cap = len(df)
    sampled = n_before_cap > cap
    if sampled:
        df = df.sample(n=cap, random_state=42).sort_index()
        st.info(f"Clustering a random sample of {cap:,} of the {n_before_cap:,} points. Raise the maximum above to use more.")
    df = df.reset_index(drop=True)

    width_km, height_km = extent_km(df["lat"], df["lon"])
    if max(width_km, height_km) > 500 and not use_far:
        st.warning(
            f"These points cover about {width_km:,.0f} km by {height_km:,.0f} km, which usually means a few points are in the wrong place "
            "(for example swapped latitude and longitude). Tick 'Also flag points far from the rest of the data' above to remove them."
        )
    else:
        st.success(f"Ready: {len(df):,} points covering about {width_km:.1f} km by {height_km:.1f} km. Open the next tab to cluster them.")

    with st.expander("Preview the data"):
        p1, p2 = st.columns([2, 3])
        p1.dataframe(df[["lat", "lon", "offence"]].head(15).rename(columns={"lat": "Latitude", "lon": "Longitude", "offence": "Offence"}), hide_index=True)
        p2.bar_chart(counts.head(15), horizontal=True)

    return Dataset(
        df=df,
        X=to_metres(df["lat"], df["lon"]),
        token=_short_hash(name, lat_col, lon_col, offence_col, flag_zero, far_km if use_far else None, sorted(chosen), cap, len(df)),
        key=_short_hash(name, lat_col, lon_col, offence_col),
        source_name=name,
        n_valid_total=n_before_cap,
        sampled=sampled,
    )
