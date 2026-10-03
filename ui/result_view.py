"""Shows one clustering result: quality scores, heatmap, cluster table and downloads."""
from __future__ import annotations

import math

import numpy as np
import streamlit as st
from streamlit_folium import st_folium

from core.dataset import Dataset
from core.summary import summarise_clusters
from ui.widgets import map_for

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
    "Points that belong to no cluster. DBSCAN, KNN and Infomap can leave points out. "
    "The three scores ignore them."
)


def fmt(value: float, digits: int) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def describe_params(params: dict) -> str:
    return ", ".join(f"{k} = {v}" for k, v in params.items())


def render_result(result: dict, ds: Dataset, map_options: dict, key_prefix: str) -> None:
    scores = result["scores"]
    st.markdown(f"**{result['algorithm']}** finished in {result['runtime_s']:.2f} s using {describe_params(result['params'])}.")

    cols = st.columns(5)
    cols[0].metric("Clusters", f"{scores['n_clusters']:,}", border=True)
    cols[1].metric("Noise", f"{scores['noise_pct']:.1f}%", help=NOISE_HELP, border=True)
    cols[2].metric("Silhouette (higher is better)", fmt(scores["silhouette"], 3), help=SILHOUETTE_HELP, border=True)
    cols[3].metric("Davies-Bouldin (lower is better)", fmt(scores["davies_bouldin"], 3), help=DAVIES_BOULDIN_HELP, border=True)
    cols[4].metric("Dunn (higher is better)", fmt(scores["dunn"], 4), help=DUNN_HELP, border=True)

    if scores["n_clusters"] == 0:
        st.warning("No clusters were found, so the heatmap is empty. Loosen the parameters, for example a larger eps or a smaller minimum cluster size.")
    elif scores["n_clusters"] < 2:
        st.warning("Only one cluster was found. The three scores need at least two, so change the parameters to split the data.")

    m = map_for(result, ds.df, map_options)
    st_folium(m, height=560, use_container_width=True, returned_objects=[], key=f"{key_prefix}_map")
    st.caption("Map tiles load from the internet. Use the layer button on the map to switch between street, light grey and satellite views.")

    summary = summarise_clusters(ds.df, result["labels"])
    st.markdown("#### Hotspots found")
    if summary.empty:
        st.write("No clusters to list.")
    else:
        st.caption("Cluster 1 is the biggest. Only the 100 biggest are listed here.")
        st.dataframe(
            summary.head(100).rename(
                columns={
                    "cluster": "Cluster", "incidents": "Incidents", "share_pct": "Share of all (%)",
                    "centre_lat": "Centre latitude", "centre_lon": "Centre longitude", "top_offences": "Most common offences",
                }
            ),
            hide_index=True,
            column_config={
                "Share of all (%)": st.column_config.NumberColumn(format="%.1f"),
                "Centre latitude": st.column_config.NumberColumn(format="%.5f"),
                "Centre longitude": st.column_config.NumberColumn(format="%.5f"),
            },
        )

    labels = np.asarray(result["labels"])
    export = ds.df.copy()
    export["cluster"] = np.where(labels >= 0, labels + 1, -1)  # -1 means noise
    left, right = st.columns(2)
    left.download_button(
        "Download clustered points (CSV)",
        export.to_csv(index=False).encode("utf-8"),
        file_name=f"clusters_{result['algorithm'].lower().replace(' ', '_')}.csv",
        mime="text/csv",
        key=f"{key_prefix}_dl_csv",
    )
    right.download_button(
        "Download map (HTML)",
        m.get_root().render().encode("utf-8"),
        file_name=f"heatmap_{result['algorithm'].lower().replace(' ', '_')}.html",
        mime="text/html",
        key=f"{key_prefix}_dl_html",
    )
