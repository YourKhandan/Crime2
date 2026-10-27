"""Shows one clustering result: quality scores, heatmap, cluster table and downloads."""
# ^ docstring: a short note about this file. python does not run it, it is only for humans.
# this file only DRAWS a result. the clustering itself was already done by make_result (core/pipeline.py).
# both tab 2 (cluster_tab.py) and tab 3 (compare_tab.py) call render_result, so the result looks the same in both.

# lets us write type hints like "str | None" without errors on older python versions.
from __future__ import annotations

# math has isnan(), which we use to test for "not a number" values.
import math

# numpy is for fast maths on arrays. "np" is the short name.
import numpy as np
# streamlit draws the website. "st" is the short name.
import streamlit as st
# st_folium is the add-on that shows a folium map inside a streamlit page.
from streamlit_folium import st_folium

# Dataset = the box from tab 1 (clean table, metre positions, token, key).
from core.dataset import Dataset
# summarise_clusters = makes the table with one row per cluster.
from core.summary import summarise_clusters
# map_for = gives us the (cached) folium map for a result. it lives in ui/widgets.py.
from ui.widgets import map_for

# the four texts below are shown as the small "?" help text on the score cards.
# they are written in CAPITALS because they are constants (they never change).
# strings inside ( ) that sit next to each other are joined by python into one long string.
SILHOUETTE_HELP = (
    "For each crime: is it closer to its own hotspot than to the next-nearest one? "
    "1 means clearly in the right place, 0 means on the border, below 0 means probably in the wrong hotspot. "
    "Higher is better."
)
DAVIES_BOULDIN_HELP = (
    "Compares how spread out each cluster is with how far it sits from its most similar neighbour. "
    "Tight clusters that are far apart score low. Lower is better, and 0 is the best possible."
)
DUNN_HELP = (
    "The gap between the two closest clusters divided by the width of the widest cluster. "
    "A high value means compact clusters with clear gaps between them. One stray point can change it a lot. "
    "Higher is better."
)
NOISE_HELP = (
    "Points that belong to no cluster. DBSCAN, Louvain and Infomap can leave points out. "
    "The three scores ignore them."
)


# turns a score into text with a fixed number of decimals. value = the score, digits = how many decimals.
def fmt(value: float, digits: int) -> str:
    # NaN means "not a number". a score is NaN when it could not be calculated (for example, with only one cluster).
    # NaN is never equal to anything, not even itself, so we cannot test it with ==. that is why we use math.isnan.
    # in that case show "n/a". otherwise, f"{value:.{digits}f}" writes the number with that many decimals,
    # for example fmt(0.70712, 3) gives "0.707".
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


# turns the parameters dictionary into one readable sentence part.
def describe_params(params: dict) -> str:
    # params looks like {"eps_m": 100.0, "min_samples": 10}.
    # .items() walks through the pairs (name, value). each pair becomes the text "eps_m = 100.0".
    # ", ".join(...) glues them together with a comma between, giving "eps_m = 100.0, min_samples = 10".
    return ", ".join(f"{k} = {v}" for k, v in params.items())


# draws everything for one result.
# result = the dictionary made by make_result (labels, scores, time, ...). ds = the Dataset.
# map_options = the map settings from the user. key_prefix = a short word ("single" or "compare")
# that keeps the widget names different between the two tabs.
def render_result(result: dict, ds: Dataset, map_options: dict, key_prefix: str) -> None:
    # the dictionary of scores: n_clusters, noise_pct, silhouette, davies_bouldin, dunn.
    scores = result["scores"]
    # one sentence on top. ** ** makes the algorithm name bold. :.2f shows the time with 2 decimals.
    st.markdown(f"**{result['algorithm']}** finished in {result['runtime_s']:.2f} s using {describe_params(result['params'])}.")

    # five equal columns in a row, one for each number card.
    cols = st.columns(5)
    # cols[0] means "the first column". .metric draws a number card: label, value, optional help text and border.
    # :, puts commas in big numbers (1200 becomes 1,200).
    cols[0].metric("Clusters", f"{scores['n_clusters']:,}", border=True)
    # the share of crimes that are noise, with 1 decimal and a % sign.
    cols[1].metric("Noise", f"{scores['noise_pct']:.1f}%", help=NOISE_HELP, border=True)
    # the three quality scores. fmt shows "n/a" if a score could not be calculated.
    cols[2].metric("Silhouette (higher is better)", fmt(scores["silhouette"], 3), help=SILHOUETTE_HELP, border=True)
    cols[3].metric("Davies-Bouldin (lower is better)", fmt(scores["davies_bouldin"], 3), help=DAVIES_BOULDIN_HELP, border=True)
    cols[4].metric("Dunn (higher is better)", fmt(scores["dunn"], 4), help=DUNN_HELP, border=True)

    # if no cluster was found at all, tell the user in a yellow box how to fix it.
    if scores["n_clusters"] == 0:
        st.warning("No clusters were found, so the heatmap is empty. Loosen the parameters, for example a larger eps or a smaller minimum cluster size.")
    # "elif" = "else if". with exactly one cluster the three scores cannot be calculated (they compare clusters).
    elif scores["n_clusters"] < 2:
        st.warning("Only one cluster was found. The three scores need at least two, so change the parameters to split the data.")

    # get the folium map for this result. it is built once and then re-used (see cached_map in widgets.py).
    m = map_for(result, ds.df, map_options)
    # show the map on the page. height=560 pixels tall. use_container_width=True makes it as wide as the page area.
    # returned_objects=[] means "do not send map clicks or zooms back to python", so moving the map never re-runs the app.
    # key gives this map a unique name.
    st_folium(m, height=560, use_container_width=True, returned_objects=[], key=f"{key_prefix}_map")
    # a small grey note under the map.
    st.caption("Map tiles load from the internet. Use the layer button on the map to switch between street, light grey and satellite views.")

    # the table with one row per cluster (size, centre, top offences).
    summary = summarise_clusters(ds.df, result["labels"])
    # a small heading.
    st.markdown("#### Hotspots found")
    # .empty is True when the table has no rows (no clusters).
    if summary.empty:
        st.write("No clusters to list.")
    else:
        # a small grey note about the numbering and the limit.
        st.caption("Cluster 1 is the biggest. Only the 100 biggest are listed here.")
        # show the table.
        st.dataframe(
            # .head(100) = the first 100 rows (the biggest clusters).
            # .rename(columns=...) changes the column headers from code names to friendly names.
            summary.head(100).rename(
                columns={
                    "cluster": "Cluster", "incidents": "Incidents", "share_pct": "Share of all (%)",
                    "centre_lat": "Centre latitude", "centre_lon": "Centre longitude", "top_offences": "Most common offences",
                }
            ),
            # hide the row numbers on the left.
            hide_index=True,
            # column_config sets how each number column is shown: "%.1f" = 1 decimal, "%.5f" = 5 decimals.
            column_config={
                "Share of all (%)": st.column_config.NumberColumn(format="%.1f"),
                "Centre latitude": st.column_config.NumberColumn(format="%.5f"),
                "Centre longitude": st.column_config.NumberColumn(format="%.5f"),
            },
        )

    # the cluster number of every crime, as a numpy array (0 = biggest cluster, -1 = noise).
    labels = np.asarray(result["labels"])
    # make a copy of the crime table, so adding a column does not change the original table.
    export = ds.df.copy()
    # add a "cluster" column. np.where(test, a, b) means: where the test is True use a, otherwise use b.
    # clustered crimes (labels >= 0) get labels + 1, so the numbers start at 1 like in the table above.
    # noise crimes get -1.
    export["cluster"] = np.where(labels >= 0, labels + 1, -1)  # -1 means noise
    # two equal columns, side by side, for the two download buttons.
    left, right = st.columns(2)
    # the first download button, in the left column.
    left.download_button(
        # the text on the button.
        "Download clustered points (CSV)",
        # the file content. .to_csv(index=False) makes csv text without the row numbers.
        # .encode("utf-8") turns the text into bytes, which the button needs.
        export.to_csv(index=False).encode("utf-8"),
        # the name of the downloaded file, for example clusters_k-means.csv.
        # .lower() makes it small letters and .replace(' ', '_') changes spaces into underscores.
        file_name=f"clusters_{result['algorithm'].lower().replace(' ', '_')}.csv",
        # tells the browser what kind of file it is.
        mime="text/csv",
        # a unique name for this button (it must differ between the two tabs).
        key=f"{key_prefix}_dl_csv",
    )
    # the second download button, in the right column. it saves the map as a standalone web page.
    right.download_button(
        "Download map (HTML)",
        # m.get_root().render() makes the full HTML text of the map. then we turn it into bytes.
        m.get_root().render().encode("utf-8"),
        file_name=f"heatmap_{result['algorithm'].lower().replace(' ', '_')}.html",
        mime="text/html",
        key=f"{key_prefix}_dl_html",
    )
