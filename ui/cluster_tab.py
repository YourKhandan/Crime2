"""Tab 2: pick one algorithm, set its parameters, see the heatmap and the scores."""
from __future__ import annotations

import streamlit as st

from algorithms import ALGORITHMS
from core.dataset import Dataset
from core.pipeline import make_result
from ui.result_view import render_result
from ui.widgets import cached_hints, render_map_options, render_params

RESULT_KEY = "single_result"


def render(ds: Dataset) -> None:
    hints = cached_hints(ds.key, ds.X)
    left, right = st.columns([1, 2.2], gap="large")

    with left:
        st.subheader("Algorithm and parameters")
        name = st.selectbox("Clustering algorithm", list(ALGORITHMS), key="single_algorithm")
        algorithm = ALGORITHMS[name]
        st.write(algorithm.DESCRIPTION)
        tip = algorithm.hint_text(hints)
        if tip:
            st.caption(tip)

        # a form means nothing runs until you press the button, however many parameters you change
        with st.form(f"single_form_{name}"):
            params = render_params(algorithm.PARAMS, f"single_{ds.key}_{name}", hints)
            run_clicked = st.form_submit_button("Run clustering", type="primary")

        with st.expander("Map display settings"):
            map_options = render_map_options("single")

    error = None
    if run_clicked:
        with right, st.spinner(f"Running {name}..."):
            try:
                st.session_state[RESULT_KEY] = make_result(name, algorithm, ds.X, ds.token, params)
            except (ValueError, RuntimeError, MemoryError) as exc:
                st.session_state.pop(RESULT_KEY, None)
                error = str(exc)

    with right:
        result = st.session_state.get(RESULT_KEY)
        if error:
            st.error(error)
        elif result and result["token"] == ds.token:
            render_result(result, ds, map_options, "single")
        else:
            if result:
                st.info("The data changed since the last run. Press Run clustering to update the results.")
            else:
                st.info("Choose an algorithm, adjust its parameters, then press Run clustering. The heatmap and scores appear here.")
