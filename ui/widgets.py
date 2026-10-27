"""Widgets shared by the tabs."""
# ^ docstring: a short note about this file. python does not run it, it is only for humans.
# this file holds the helpers that BOTH tab 2 and tab 3 use:
# the parameter widgets, the map settings, and the remembered (cached) hints and maps.

# lets us write type hints like "list[Param]" without errors on older python versions.
from __future__ import annotations

# json turns python data (like a dictionary) into text, and back. we use it to make a "name tag" for the map settings.
import json

# numpy is for fast maths on arrays. "np" is the short name.
import numpy as np
# streamlit draws the website. "st" is the short name.
import streamlit as st

# Param is the "description card" for one setting (defined in algorithms/base.py).
from algorithms.base import Param
# HEAT_MODES = the three heat styles. build_map = the function that builds the folium map. both from core/heatmap.py.
from core.heatmap import HEAT_MODES, build_map
# compute_hints = works out starting values from the data, for example a good eps for DBSCAN.
from core.summary import compute_hints


# @st.cache_data means "remember the answer". streamlit re-runs the file on every click,
# so without this the hints would be worked out again each time.
# show_spinner=False hides the "running" message.
@st.cache_data(show_spinner=False)
# dataset_key = the short fingerprint of the file and column choice (from tab 1). _X = the metre positions.
# the _ in front of _X tells streamlit: "do NOT use this one to decide if the answer is already remembered".
# checking a huge array would be slow, so only dataset_key is used. same file and columns = same remembered answer.
def cached_hints(dataset_key: str, _X: np.ndarray) -> dict:
    """Starting values worked out from the data (cached once per file and column choice)."""
    # the real work is done in core/summary.py.
    return compute_hints(_X)


# draws one widget for every setting of an algorithm, and gives back the chosen values.
# params = the list of Param cards. key_prefix = text put in front of every widget name to keep the names unique.
# hints = the starting values worked out from the data. it gives back a dictionary like {"eps_m": 100.0, ...}.
def render_params(params: list[Param], key_prefix: str, hints: dict) -> dict:
    """Draw one widget per Param and return {param name: chosen value}."""
    # an empty dictionary. we will fill it with the chosen value of each setting.
    values: dict = {}
    # go through the Param cards one by one. p is the current card.
    for p in params:
        # the unique name of this widget, for example "single_ab12cd34ef_DBSCAN__eps_m".
        # streamlit needs unique names so it can remember each widget separately.
        key = f"{key_prefix}__{p.name}"
        # the starting value. if the card has a hint name (like "eps_m"), take the value from hints.
        # hints.get(name, fallback) gives the fallback (p.default) if the name is not in the dictionary.
        # if the card has no hint name, just use p.default.
        default = hints.get(p.hint, p.default) if p.hint else p.default

        # kind "int" = a slider that gives whole numbers.
        if p.kind == "int":
            # force the starting value to stay between min and max (max() lifts it up to min, min() pushes it down to max).
            # streamlit shows an error if the starting value is outside the range. int() makes it a whole number.
            default = int(min(max(default, p.min), p.max))
            # draw the slider: label, smallest, biggest, starting value. step = jump size (use 1 if no step is given).
            # key = its unique name. help = the "?" text. st.slider gives back the value the user chose.
            values[p.name] = st.slider(p.label, int(p.min), int(p.max), default, step=int(p.step or 1), key=key, help=p.help)
        # kind "int_input" = a number box for whole numbers (used for the random seed).
        elif p.kind == "int_input":
            default = int(min(max(default, p.min), p.max))
            # st.number_input draws the box. int(...) makes sure the answer is a whole number.
            values[p.name] = int(st.number_input(p.label, int(p.min), int(p.max), default, step=int(p.step or 1), key=key, help=p.help))
        # kind "float" = a number box that allows decimals (like eps in metres).
        elif p.kind == "float":
            # same idea, but with decimal numbers.
            default = float(min(max(default, p.min), p.max))
            # format="%g" shows the number in a tidy way (300.0 is shown as 300 and 0.5 as 0.5).
            values[p.name] = float(
                st.number_input(p.label, float(p.min), float(p.max), default, step=float(p.step or 0.1), format="%g", key=key, help=p.help)
            )
        # kind "select" = a dropdown.
        elif p.kind == "select":
            # make a normal list of the choices.
            options = list(p.options)
            # index = which choice is selected at the start. options.index(default) finds the position of the default in the list.
            values[p.name] = st.selectbox(p.label, options, index=options.index(default), key=key, help=p.help)
        # kind "bool" = a tick box (True or False).
        elif p.kind == "bool":
            # value = ticked or not at the start. bool(...) makes sure it is True or False.
            values[p.name] = st.checkbox(p.label, value=bool(default), key=key, help=p.help)
        # any other kind word is a mistake in an algorithm file, so stop with a clear message.
        else:
            raise ValueError(f"Unknown parameter kind: {p.kind}")
    # give back everything the user chose, for example {"eps_m": 300.0, "min_samples": 10, "metric": "euclidean"}.
    return values


# draws the map settings. prefix = "single" or "compare", to keep the widget names different in the two tabs.
# it gives back a dictionary. the names inside it are exactly the names build_map() expects.
def render_map_options(prefix: str) -> dict:
    """Heatmap and layer settings. Returns keyword arguments for build_map()."""
    # radio buttons. list(HEAT_MODES) gives the three texts the user sees. mode_label is the text they picked.
    mode_label = st.radio(
        "What makes a spot hot?",
        list(HEAT_MODES),
        key=f"{prefix}_heat_mode",
        help="Cluster size: points in bigger clusters glow hotter and noise is hidden. "
        "Equal: every clustered point counts the same. Plain density: ignores the clustering, "
        "so you can compare it with the cluster-based views.",
    )
    # a slider for the size of the glow. the numbers are: smallest 5, biggest 50, starting value 15.
    radius = st.slider("Heat radius", 5, 50, 15, key=f"{prefix}_radius")
    # a slider for how soft the glow edges are: smallest 5, biggest 40, start 12.
    blur = st.slider("Heat blur", 5, 40, 12, key=f"{prefix}_blur")
    # a slider for the faintest heat: smallest 0.0, biggest 1.0, start 0.3, jump size 0.05.
    min_opacity = st.slider("Faintest heat", 0.0, 1.0, 0.3, 0.05, key=f"{prefix}_min_opacity")
    # build the dictionary to give back. the tick boxes are drawn right here, while the dictionary is being built.
    # each tick box gives True or False. the text before ":" is the name that build_map expects.
    return {
        # HEAT_MODES[mode_label] changes the text the user picked into the short code (like "cluster_size").
        "heat_mode": HEAT_MODES[mode_label],
        "radius": radius,
        "blur": blur,
        "min_opacity": min_opacity,
        # the number after the label is the starting state: False = not ticked, True = ticked.
        "include_noise": st.checkbox("Show noise points faintly in the heatmap", False, key=f"{prefix}_noise"),
        "show_centres": st.checkbox("Mark cluster centres", True, key=f"{prefix}_centres"),
        "show_hulls": st.checkbox("Outline clusters", False, key=f"{prefix}_hulls"),
        "show_points": st.checkbox("Show incident dots", False, key=f"{prefix}_points"),
    }


# st.cache_resource also means "remember the answer", but it remembers the real object itself (not a copy).
# that is what we need for a map object. max_entries=12 means: keep at most 12 maps, and throw away the oldest ones.
# show_spinner="..." shows that message while a new map is being built.
@st.cache_resource(show_spinner="Drawing the map...", max_entries=12)
# only result_id and options_json decide if a map is already remembered. the ones with _ in front are ignored,
# because they are big (the table and the labels) and are already represented by result_id.
# result_id = the unique id of this clustering result. options_json = the map settings written as text.
def cached_map(result_id: str, options_json: str, _df, _labels, _options: dict):
    """Build the folium map once per result and set of display options.

    Re-using the same map object keeps its zoom and position when you touch other widgets.
    """
    # **_options "unpacks" the dictionary into named inputs. {"radius": 15, "blur": 12} becomes radius=15, blur=12.
    # this works because render_map_options uses exactly the names build_map expects.
    return build_map(_df, _labels, **_options)


# the function the other files call to get a map. it fills in the "name tag" for the cache.
def map_for(result: dict, df, options: dict):
    # json.dumps turns the options dictionary into text, because the cache needs a simple text to compare.
    # sort_keys=True puts the names in a fixed order, so the same settings always give the same text.
    # so: same result + same settings = same map from the cache. change a setting and a new map is built.
    return cached_map(result["id"], json.dumps(options, sort_keys=True), df, result["labels"], options)
