"""Widgets shared by the tabs."""
from __future__ import annotations

import json

import numpy as np
import streamlit as st

from algorithms.base import Param
from core.heatmap import HEAT_MODES, build_map
from core.summary import compute_hints


@st.cache_data(show_spinner=False)
def cached_hints(dataset_key: str, _X: np.ndarray) -> dict:
    """Starting values worked out from the data (cached once per file and column choice)."""
    return compute_hints(_X)


def render_params(params: list[Param], key_prefix: str, hints: dict) -> dict:
    """Draw one widget per Param and return {param name: chosen value}."""
    values: dict = {}
    for p in params:
        key = f"{key_prefix}__{p.name}"
        default = hints.get(p.hint, p.default) if p.hint else p.default

        if p.kind == "int":
            default = int(min(max(default, p.min), p.max))
            values[p.name] = st.slider(p.label, int(p.min), int(p.max), default, step=int(p.step or 1), key=key, help=p.help)
        elif p.kind == "int_input":
            default = int(min(max(default, p.min), p.max))
            values[p.name] = int(st.number_input(p.label, int(p.min), int(p.max), default, step=int(p.step or 1), key=key, help=p.help))
        elif p.kind == "float":
            default = float(min(max(default, p.min), p.max))
            values[p.name] = float(
                st.number_input(p.label, float(p.min), float(p.max), default, step=float(p.step or 0.1), format="%g", key=key, help=p.help)
            )
        elif p.kind == "select":
            options = list(p.options)
            values[p.name] = st.selectbox(p.label, options, index=options.index(default), key=key, help=p.help)
        elif p.kind == "bool":
            values[p.name] = st.checkbox(p.label, value=bool(default), key=key, help=p.help)
        else:
            raise ValueError(f"Unknown parameter kind: {p.kind}")
    return values


def render_map_options(prefix: str) -> dict:
    """Heatmap and layer settings. Returns keyword arguments for build_map()."""
    mode_label = st.radio(
        "What makes a spot hot?",
        list(HEAT_MODES),
        key=f"{prefix}_heat_mode",
        help="Cluster size: points in bigger clusters glow hotter and noise is hidden. "
        "Equal: every clustered point counts the same. Plain density: ignores the clustering, "
        "so you can compare it with the cluster-based views.",
    )
    radius = st.slider("Heat radius", 5, 50, 15, key=f"{prefix}_radius")
    blur = st.slider("Heat blur", 5, 40, 12, key=f"{prefix}_blur")
    min_opacity = st.slider("Faintest heat", 0.0, 1.0, 0.3, 0.05, key=f"{prefix}_min_opacity")
    return {
        "heat_mode": HEAT_MODES[mode_label],
        "radius": radius,
        "blur": blur,
        "min_opacity": min_opacity,
        "include_noise": st.checkbox("Show noise points faintly in the heatmap", False, key=f"{prefix}_noise"),
        "show_centres": st.checkbox("Mark cluster centres", True, key=f"{prefix}_centres"),
        "show_hulls": st.checkbox("Outline clusters", False, key=f"{prefix}_hulls"),
        "show_points": st.checkbox("Show incident dots", False, key=f"{prefix}_points"),
    }


@st.cache_resource(show_spinner="Drawing the map...", max_entries=12)
def cached_map(result_id: str, options_json: str, _df, _labels, _options: dict):
    """Build the folium map once per result and set of display options.

    Re-using the same map object keeps its zoom and position when you touch other widgets.
    """
    return build_map(_df, _labels, **_options)


def map_for(result: dict, df, options: dict):
    return cached_map(result["id"], json.dumps(options, sort_keys=True), df, result["labels"], options)
