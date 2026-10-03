"""Builds the interactive map: a heatmap driven by the clusters, on real map tiles."""
from __future__ import annotations

import html

import folium
import numpy as np
import pandas as pd
from folium.plugins import Fullscreen, HeatMap
from scipy.spatial import ConvexHull

from .summary import summarise_clusters

PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#e377c2", "#8c564b",
    "#bcbd22", "#393b79", "#ad494a", "#637939", "#7b4173", "#e7ba52", "#3182bd", "#843c39",
]
NOISE_COLOUR = "#8a8f98"

HEAT_MODES = {
    "Cluster size (bigger hotspot = hotter)": "cluster_size",
    "Equal (shows where the clusters are)": "equal",
    "Plain density (ignores the clustering)": "density",
}


def cluster_colour(cluster_index: int) -> str:
    return PALETTE[int(cluster_index) % len(PALETTE)]


def _heat_weights(labels: np.ndarray, mode: str, include_noise: bool) -> np.ndarray:
    clustered = labels >= 0
    if mode == "density":
        return np.ones(len(labels))
    weights = np.zeros(len(labels))
    if clustered.any():
        if mode == "cluster_size":
            sizes = np.bincount(labels[clustered])
            # square root keeps small clusters visible next to a huge one
            weights[clustered] = np.sqrt(sizes[labels[clustered]] / sizes.max())
            weights[clustered] = np.clip(weights[clustered], 0.2, 1.0)
        else:
            weights[clustered] = 1.0
    if include_noise:
        weights[~clustered] = 0.1
    return weights


def build_map(
    df: pd.DataFrame,
    labels: np.ndarray,
    *,
    heat_mode: str = "cluster_size",
    radius: int = 15,
    blur: int = 12,
    min_opacity: float = 0.3,
    include_noise: bool = False,
    show_centres: bool = True,
    show_hulls: bool = False,
    show_points: bool = False,
    max_points: int = 1500,
    max_clusters_drawn: int = 100,
) -> folium.Map:
    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)
    labels = np.asarray(labels)
    n = len(lat)

    m = folium.Map(location=[float(np.median(lat)), float(np.median(lon))], zoom_start=12, tiles=None, control_scale=True, prefer_canvas=True)
    folium.TileLayer("OpenStreetMap", name="Street map", show=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Light grey map", max_zoom=16, show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Satellite", max_zoom=19, show=False,
    ).add_to(m)

    # zoom to where most of the data is, ignoring a few far-away stragglers
    south, north = np.percentile(lat, [1, 99])
    west, east = np.percentile(lon, [1, 99])
    if north - south < 1e-3:
        south, north = south - 0.005, north + 0.005
    if east - west < 1e-3:
        west, east = west - 0.005, east + 0.005
    m.fit_bounds([[float(south), float(west)], [float(north), float(east)]], padding=(20, 20))

    # heatmap
    weights = _heat_weights(labels, heat_mode, include_noise)
    visible = weights > 0
    if visible.any():
        heat_data = np.column_stack([lat[visible], lon[visible], weights[visible]]).round(5).tolist()
        HeatMap(heat_data, name="Heatmap", radius=int(radius), blur=int(blur), min_opacity=float(min_opacity)).add_to(m)

    summary = summarise_clusters(df, labels)
    drawn = summary.head(max_clusters_drawn)

    if show_hulls and len(drawn):
        group = folium.FeatureGroup(name="Cluster outlines", show=True)
        for row in drawn.itertuples():
            idx = np.flatnonzero(labels == row.cluster - 1)
            if len(idx) < 3:
                continue
            pts = np.column_stack([lat[idx], lon[idx]])
            try:
                outline = pts[ConvexHull(pts).vertices].tolist()
            except Exception:  # all points on one line, etc.
                continue
            colour = cluster_colour(row.cluster - 1)
            folium.Polygon(
                outline, color=colour, weight=2, fill=True, fill_opacity=0.10,
                tooltip=f"Cluster {row.cluster}: {row.incidents:,} incidents",
            ).add_to(group)
        group.add_to(m)

    if show_points:
        group = folium.FeatureGroup(name="Incident dots", show=True)
        rng = np.random.default_rng(0)
        for i in rng.choice(n, size=min(n, int(max_points)), replace=False):
            colour = NOISE_COLOUR if labels[i] < 0 else cluster_colour(labels[i])
            where = "noise" if labels[i] < 0 else f"cluster {labels[i] + 1}"
            folium.CircleMarker(
                [lat[i], lon[i]], radius=3, color=colour, weight=0, fill=True, fill_color=colour, fill_opacity=0.75,
                tooltip=f"{html.escape(str(df['offence'].iloc[i]))} ({where})",
            ).add_to(group)
        group.add_to(m)

    if show_centres and len(drawn):
        group = folium.FeatureGroup(name="Cluster centres", show=True)
        biggest = float(drawn["incidents"].max())
        for row in drawn.itertuples():
            colour = cluster_colour(row.cluster - 1)
            popup = (
                f"<b>Cluster {row.cluster}</b><br>{row.incidents:,} incidents ({row.share_pct:.1f}% of all)<br>"
                f"Top offences: {html.escape(str(row.top_offences))}"
            )
            folium.CircleMarker(
                [row.centre_lat, row.centre_lon],
                radius=6 + 12 * float(np.sqrt(row.incidents / biggest)),
                color="#ffffff", weight=2, fill=True, fill_color=colour, fill_opacity=0.9,
                tooltip=f"Cluster {row.cluster}: {row.incidents:,} incidents",
                popup=folium.Popup(popup, max_width=300),
            ).add_to(group)
        group.add_to(m)

    Fullscreen().add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m
