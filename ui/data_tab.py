"""Tab 1: load a file, check every latitude/longitude, choose what to cluster."""
# ^ the text in triple quotes above is a "docstring". it is a short note about what this file does.

# this line lets us write type hints like "Dataset | None" without errors on older python versions.
from __future__ import annotations

# hashlib can turn any text into a short "fingerprint" (called a hash).
# we use it below to make the "token" and "key" for our data.
import hashlib

# Path helps us work with file locations. it works the same on windows, mac and linux.
from pathlib import Path
# so for uploading from user it helps
import pandas as pd

# streamlit is the library that draws the website. "st" is alias.
# every st.something(...) line puts something on the web page.
import streamlit as st

# these come from our own file core/data_io.py:
# guess_columns = guesses which column is latitude / longitude / offence
# read_columns = reads ONLY the columns we ask for
# read_header = reads only the first row (the column names)
from core.data_io import guess_columns, read_columns, read_header

# Dataset is a small "box" (a class) that holds the cleaned data so we can hand it to the other tabs.
from core.dataset import Dataset

# extent_km = tells how many km wide and tall the data is
# to_metres = changes latitude/longitude (degrees) into metres
from core.geo import extent_km, to_metres

# validate_coordinates = checks every latitude and longitude
# ValidationReport = the "result box" it gives back. we only use it here as a type hint.
from core.validation import ValidationReport, validate_coordinates

# the path of the sample csv file.
# __file__ is this file (data_tab.py). resolve() makes it a full path.
# .parent = the ui folder, .parent.parent = the main project folder.
# the "/" at the end just adds the file name to that folder path.
SAMPLE_PATH = Path(__file__).resolve().parent.parent / "sample_crime_data.csv"
# the text shown in the offence dropdown when the user does not want any offence column.
# these names are written in CAPITALS because they are constants (they never change).
NO_COLUMN = "(none)"
# radio button choice 1: throw away the bad rows and keep going.
DROP = "Drop them and continue"
# radio button choice 2: stop everything if there are bad rows.
STOP = "Stop so I can fix the file"


# streamlit runs this whole file again on every click, so without this the header would be re-read every time.
# if the same input comes again, the saved answer is used. show_spinner=False hides the "running" message. 
# @st.cache_data means "remember the answer".
@st.cache_data(show_spinner=False)
# raw = the file's content as bytes, name = the file name. "-> list[str]" means it gives back a list of text.
# the _ at the start of the name is a habit that means "helper function, only used inside this file".
def _header(raw: bytes, name: str) -> list[str]:
    # the real work is done by read_header in core/data_io.py. we just call it.
    return read_header(raw, name)


# same idea: remember the answer so the file is not read and checked again on every click.
# the spinner text is shown to the user while it works.
@st.cache_data(show_spinner="Reading and checking the file...")
# this takes the file and the user's choices, and gives back a ValidationReport.
# "str | None" means the value can be text or None (nothing).
def _load_and_validate(raw: bytes, name: str, lat_col: str, lon_col: str, offence_col: str | None, flag_zero: bool, far_km: float | None) -> ValidationReport:
    # make a list of the columns we need. start with latitude and longitude.
    # then add the offence column only if the user picked one. if not, we add an empty list [] (adds nothing).
    columns = [lat_col, lon_col] + ([offence_col] if offence_col else [])
    # read the file but ONLY these columns. every other column in the file is skipped.
    df = read_columns(raw, name, columns)  # only these columns are read
    # check every row. flag_zero = should (0,0) count as bad. far_km = how far from the centre is "too far".
    return validate_coordinates(df, lat_col, lon_col, offence_col, flag_zero_zero=flag_zero, max_km_from_centre=far_km)


# makes a short fingerprint from any number of things (*parts means "take as many inputs as you want").
def _short_hash(*parts) -> str:
    # step by step:
    # map(str, parts) = turn every part into text first (some parts are numbers, True/False or None)
    # "|".join(...) = glue them into one text with | between them, e.g. "delhi|1000|None"
    # .encode() = change the text into bytes, because hashlib.md5 only accepts bytes
    # hashlib.md5(...) = make the fingerprint
    # .hexdigest() = write the fingerprint as normal text (letters a-f and numbers)
    # [:10] = keep only the first 10 characters
    # same inputs always give the same fingerprint. if anything changes, the fingerprint changes.
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:10]


# this function draws the whole first tab. it gives back a Dataset, or None if the data is not ready.
def render() -> Dataset | None:
    # a heading at the top of the tab.
    st.subheader("Load your data")
    # make two columns side by side. [3, 2] means widths in the ratio 3 : 2 (left is bigger).
    left, right = st.columns([3, 2], gap="large")
    # gap="large" puts more space between them. left and right are the two columns.
    # everything indented under "with left:" is drawn inside the left column.
    with left:
        # the upload box. type=[...] allows only these file types.
        # key is a unique name so streamlit can remember this widget between runs.
        # upload is None until the user picks a file, after that it holds the file.
        upload = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"], key="upload")
    # everything under this is drawn in the right column.
    with right:
        # st.write just prints text on the page.
        st.write(
            "Each row needs a latitude, a longitude and an offence. Every other column in your file is ignored."
        )
        # a tick box. use_sample is True when it is ticked and False when it is not.
        use_sample = st.checkbox("Use the built-in sample data instead (synthetic, with a few bad rows)", key="use_sample")

    # case 1: the user uploaded a file.
    if upload is not None:
        # raw = the file's content (bytes). name = the file's name.
        raw, name = upload.getvalue(), upload.name
    # case 2: nothing uploaded, but the sample box is ticked and the sample file really exists.
    elif use_sample and SAMPLE_PATH.exists():
        # read the sample file from the project folder instead.
        raw, name = SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name
    # case 3: no file at all.
    else:
        # show a blue message telling the user what to do.
        st.info("Upload a file above, or tick the sample data option, to begin.")
        # stop the function here and give back nothing. app.py then tells the other tabs to wait.
        return None

    # try to do something risky (reading a file that might be broken).
    try:
        # get the list of column names from the file.
        columns = _header(raw, name)
    # if anything goes wrong, catch the error instead of crashing the app. exc holds the error message.
    except Exception as exc:  # unreadable or empty file
        # show a red error box that says what went wrong.
        st.error(f"Could not read {name}: {exc}. Check that it is a CSV or .xlsx file with a header row.")
        # stop here.
        return None
    # we need at least two columns (latitude and longitude).
    if len(columns) < 2:
        # show a red error and stop.
        st.error("The file has fewer than two columns. It needs at least latitude and longitude.")
        return None

    # guess which columns are latitude, longitude and offence from their names.
    # guess is a dictionary like {"lat": "Latitude", "lon": "Longitude", "offence": "Offence"}.
    # a value can be None if nothing matched.
    guess = guess_columns(columns)

    # "####" in markdown makes a small heading.
    st.markdown("#### Choose the columns")
    # three equal columns for the three dropdowns.
    c1, c2, c3 = st.columns(3)
    # a dropdown inside the first column. its options are all the column names.
    # index = which option is selected at the start: the guessed one if we have a guess, else the first (0).
    # key has the file name in it, so a new file gets fresh dropdowns.
    # lat_col is the column name the user picked.
    lat_col = c1.selectbox("Latitude column", columns, index=columns.index(guess["lat"]) if guess["lat"] else 0, key=f"lat_{name}")
    # same for longitude. if we have no guess we start on the second column (index 1),
    # or the last one if the file only has one column. min(1, ...) makes sure the number is never too big.
    lon_col = c2.selectbox("Longitude column", columns, index=columns.index(guess["lon"]) if guess["lon"] else min(1, len(columns) - 1), key=f"lon_{name}")
    # the offence dropdown gets one extra option, "(none)", put at the front.
    offence_options = [NO_COLUMN] + columns
    # the offence dropdown. if there is no guess it starts on "(none)" (index 0).
    offence_choice = c3.selectbox(
        "Offence column", offence_options, index=offence_options.index(guess["offence"]) if guess["offence"] else 0, key=f"offence_{name}"
    )
    # change "(none)" into a real Python None. otherwise keep the column name.
    offence_col = None if offence_choice == NO_COLUMN else offence_choice
    # make sure the user did not pick the same column twice.
    # {...} is a set, and a set removes repeats. so if all picks are different, the set has 3 items.
    # when there is no offence column we put "-" in its place so we still count 3 items.
    # if the set has fewer than 3 items, at least two picks are the same column.
    if len({lat_col, lon_col, offence_col or "-"}) < 3:
        # show a red error and stop.
        st.error("Pick a different column for each of latitude, longitude and offence.")
        return None

    # small heading for the checks section.
    st.markdown("#### Coordinate checks")
    # a box with a border to hold the check settings.
    with st.container(border=True):
        # a short text that tells which checks always happen.
        st.write(
            "Always checked: latitude and longitude must be numbers, latitude must be between -90 and 90, "
            "and longitude must be between -180 and 180."
        )
        # two columns inside the box. a is the left one, b is the right one.
        a, b = st.columns(2)
        # tick box in the left column. the True after the label means it starts ticked.
        # help= is the small "?" text that shows when you hover.
        flag_zero = a.checkbox("Treat exactly 0, 0 as invalid", True, key="flag_zero", help="0, 0 is usually what software writes when the real location is unknown.")
        # radio buttons: the user picks DROP or STOP. policy holds the text of the chosen one.
        policy = a.radio("If invalid coordinates are found", [DROP, STOP], key="policy")
        # tick box in the right column. it starts un-ticked (False).
        use_far = b.checkbox("Also flag points far from the rest of the data", False, key="use_far", help="Catches points that are valid on Earth but clearly in the wrong place, such as another country.")
        # a number box. the numbers are: smallest allowed 1, biggest allowed 20000, start value 100, +/- step of 10.
        # disabled=not use_far means it is greyed out unless the tick box above is ticked.
        far_km = b.number_input("Flag points farther than this from the centre of the data (km)", 1.0, 20000.0, 100.0, step=10.0, key="far_km", disabled=not use_far)

    # try to read and check the file.
    try:
        # float(far_km) if use_far else None = only send the distance if the user ticked the box.
        # sending None means "do not do the far-away check".
        report = _load_and_validate(raw, name, lat_col, lon_col, offence_col, flag_zero, float(far_km) if use_far else None)
    # if something goes wrong (for example a column that is not what we expected), show the error.
    except Exception as exc:
        st.error(f"Could not read those columns: {exc}")
        return None

    # four equal columns for the four number cards.
    m1, m2, m3, m4 = st.columns(4)
    # a "metric" is a number card with a label. f"{report.n_rows:,}" adds commas, so 3250 shows as 3,250.
    # border=True draws a box around the card.
    m1.metric("Rows in file", f"{report.n_rows:,}", border=True)
    # how many rows passed all the checks.
    m2.metric("Valid rows", f"{report.n_valid:,}", border=True)
    # how many rows failed at least one check.
    m3.metric("Invalid rows", f"{report.n_invalid:,}", border=True)
    # .nunique() counts how many DIFFERENT offence names there are.
    m4.metric("Offence types", f"{report.clean['offence'].nunique():,}", border=True)

    # if no row is bad, show a green success message.
    if report.n_invalid == 0:
        st.success(f"All {report.n_rows:,} latitude and longitude pairs are valid.")
    # otherwise there are bad rows.
    else:
        # build the message text first, so we can reuse it below.
        message = f"{report.n_invalid:,} of {report.n_rows:,} rows have invalid coordinates."
        # if the user chose STOP, show a red error.
        if policy == STOP:
            st.error(message + " Fix them in your file and upload it again, or switch the option above to drop them.")
        # if the user chose DROP, show a yellow warning.
        else:
            st.warning(message + " They are left out of the analysis.")
        # an expander is a box you can click to open or close.
        # expanded=policy == STOP means: open by default only when the policy is STOP.
        with st.expander("See the invalid rows", expanded=policy == STOP):
            # explain why the counts can add up to more than the number of bad rows.
            st.write("A row can have more than one problem, so the counts below can add up to more than the number of rows.")
            # problem_counts is a dictionary like {"Latitude is outside -90 to 90": 2, ...}.
            # list(dict) gives the keys (the problems) and .values() gives the counts.
            # we put them in a 2-column table. hide_index=True hides the row numbers on the left.
            st.dataframe(pd.DataFrame({"Problem": list(report.problem_counts), "Rows": list(report.problem_counts.values())}), hide_index=True)
            # show the first 500 bad rows. rename() gives the "file_row" column a nicer name.
            st.dataframe(report.invalid.head(500).rename(columns={"file_row": "Row in file"}), hide_index=True)
            # a small grey note under the table.
            st.caption("Row numbers count the header as row 1. The table shows the first 500 invalid rows; the download has all of them.")
            # a download button. the pieces are: button text, the data, the file name, the file type, and a key.
            # .to_csv(index=False) turns the table into csv text without the row numbers.
            # .encode("utf-8") turns that text into bytes, which the button needs.
            st.download_button("Download invalid rows (CSV)", report.invalid.to_csv(index=False).encode("utf-8"), file_name="invalid_rows.csv", mime="text/csv", key="dl_invalid")
        # if the user chose STOP, do not go any further.
        if policy == STOP:
            return None

    # we need at least 10 good rows, otherwise clustering makes no sense.
    if report.n_valid < 10:
        st.error("Fewer than 10 valid rows are left, which is too few to cluster.")
        return None

    # heading for the filter section.
    st.markdown("#### Choose what to cluster")
    # .value_counts() counts how many times each offence appears, most common first.
    # counts looks like: THEFT 1200, ASSAULT 800, ...
    counts = report.clean["offence"].value_counts()
    # two columns with widths 3 : 1, so f1 gets 3 parts (75%) and f2 gets 1 part (25%).
    f1, f2 = st.columns([3, 1])
    # a box where the user can pick many offence types. the options are the offence names (counts.index).
    # format_func only changes how an option is SHOWN, not its real value.
    # "lambda o: ..." is a tiny one-line function. it turns "THEFT" into "THEFT (1,200)".
    # chosen is a list of the picked offences. it is an empty list if nothing is picked.
    chosen = f1.multiselect(
        "Offence types (leave empty to use all)",
        list(counts.index),
        format_func=lambda o: f"{o} ({counts[o]:,})",
        key=f"offence_filter_{name}_{offence_col}",
    )
    # a number box for the most points we will cluster. numbers: smallest 1000, biggest 500,000, start 30,000.
    # 500_000 is the same as 500000, the _ only makes it easier to read.
    # int(...) makes sure we get a whole number back.
    cap = int(f2.number_input("Maximum points to cluster", 1000, 500_000, 30_000, step=1000, key="max_points", help="Bigger data is sampled at random to keep the app responsive."))
    # keep only the rows whose offence is in the chosen list.
    # .isin(chosen) gives True or False for every row, and putting that inside [ ] keeps only the True rows.
    # if nothing was chosen (empty list counts as False), we keep all the clean rows.
    df = report.clean[report.clean["offence"].isin(chosen)] if chosen else report.clean
    # if the filter left us with no rows at all, tell the user and stop.
    if df.empty:
        st.error("No rows match that offence selection.")
        return None
    # remember how many rows we have after the offence filter, before any cutting down.
    n_before_cap = len(df)
    # True if we have more rows than the limit.
    sampled = n_before_cap > cap
    # if we have too many rows, pick a random smaller set.
    if sampled:
        # n=cap means pick that many rows at random.
        # random_state=42 fixes the randomness, so we get the same rows every time.
        # sort_index() puts the picked rows back in their original order.
        df = df.sample(n=cap, random_state=42).sort_index()
        # tell the user we used a sample.
        st.info(f"Clustering a random sample of {cap:,} of the {n_before_cap:,} points. Raise the maximum above to use more.")
    # give the rows fresh numbers 0, 1, 2, ... and throw away the old numbers (drop=True).
    # this matters because the clustering results are matched to the rows by position.
    df = df.reset_index(drop=True)

    # how wide and how tall (in km) is the area that the points cover.
    width_km, height_km = extent_km(df["lat"], df["lon"])
    # if the area is bigger than 500 km and the "far points" check is off, something is probably wrong.
    if max(width_km, height_km) > 500 and not use_far:
        # warn the user and suggest turning on the far-points check.
        st.warning(
            f"These points cover about {width_km:,.0f} km by {height_km:,.0f} km, which usually means a few points are in the wrong place "
            "(for example swapped latitude and longitude). Tick 'Also flag points far from the rest of the data' above to remove them."
        )
    # otherwise everything looks fine.
    else:
        # green message with the number of points and the size of the area.
        st.success(f"Ready: {len(df):,} points covering about {width_km:.1f} km by {height_km:.1f} km. Open the next tab to cluster them.")

    # a click-to-open box that shows a preview of the data.
    with st.expander("Preview the data"):
        # two columns. the left (2 parts) holds the table and the right (3 parts) holds the chart.
        p1, p2 = st.columns([2, 3])
        # pick the 3 columns we care about, take the first 15 rows, and give the headers nicer names.
        p1.dataframe(df[["lat", "lon", "offence"]].head(15).rename(columns={"lat": "Latitude", "lon": "Longitude", "offence": "Offence"}), hide_index=True)
        # a bar chart of the 15 most common offences. horizontal=True lays the bars sideways.
        p2.bar_chart(counts.head(15), horizontal=True)

    # everything worked, so pack the results in a Dataset box and give it back to app.py.
    return Dataset(
        # the clean table with the columns lat, lon, offence, file_row.
        df=df,
        # the same points, but as x/y positions in metres. the clustering algorithms use this one.
        X=to_metres(df["lat"], df["lon"]),
        # a fingerprint of everything that decides WHICH data we are looking at.
        # if any of these change, the token changes, and old results are marked as out of date.
        # far_km only counts if the far check is on. sorted(chosen) keeps the order the same every time,
        # because the multiselect remembers the order the user clicked in.
        token=_short_hash(name, lat_col, lon_col, offence_col, flag_zero, far_km if use_far else None, sorted(chosen), cap, len(df)),
        # a shorter fingerprint of only the file and the column choices.
        # it stays the same when the user just changes the offence filter, so their parameter settings are not reset.
        key=_short_hash(name, lat_col, lon_col, offence_col),
        # the file name, so we can show it later.
        source_name=name,
        # how many valid rows were left after the offence filter, before the point limit was applied.
        n_valid_total=n_before_cap,
        # True if we had to take a random sample.
        sampled=sampled,
    )