"""Tab 2: pick one algorithm, set its parameters, see the heatmap and the scores."""

# lets us write type hints like "Dataset | None" without errors on older python versions.
from __future__ import annotations

import streamlit as st

# ALGORITHMS is a dictionary from algorithms/__init__.py. it looks like this:
# {"K-Means": kmeans_algo, "DBSCAN": dbscan_algo, "KNN": knn_algo, "Infomap": infomap_algo}
# the key is the name shown to the user, the value is the python file (module) of that algorithm.
from algorithms import ALGORITHMS

# Dataset is the "box" from tab 1. it holds the clean table (df), the metre positions (X), a token and a key.
from core.dataset import Dataset

# make_result runs one algorithm, measures the time, and calculates the three scores.
# it gives everything back in one dictionary.
from core.pipeline import make_result

# render_result draws the scores, the map, the hotspot table and the download buttons.
from ui.result_view import render_result

# cached_hints = gets a suggested starting value from the data, for example a good eps for DBSCAN
# render_map_options = draws the map settings (heat radius, blur and so on)
# render_params = draws one widget for every parameter of an algorithm
from ui.widgets import cached_hints, render_map_options, render_params

# the name we use to save the result inside st.session_state.
# session_state is like a notebook that streamlit keeps while the user stays on the page.
# we need it because streamlit re-runs this whole file on every click, and normal variables would be forgotten.
# it is written in CAPITALS because it is a constant (it never changes).
RESULT_KEY = "single_result"


# this function draws the whole second tab. ds is the Dataset made in tab 1. it gives back nothing (None).
def render(ds: Dataset) -> None:
    # ask for hints made from the data. for example, a suggested eps (walking distance) for DBSCAN.
    # it is cached, so it is worked out once per file and not on every click.
    hints = cached_hints(ds.key, ds.X)
    # make two columns side by side. [1, 2.2] means the right one is 2.2 times wider than the left one.
    # the left is for settings, the right is for results. gap="large" adds more space between them.
    left, right = st.columns([1, 2.2], gap="large")

    # everything indented under "with left:" is drawn in the left column.
    with left:
        # a heading at the top of the left column.
        st.subheader("Algorithm and parameters")
        # a dropdown to choose the algorithm. list(ALGORITHMS) gives just the names (the dictionary keys).
        # key is a unique name so streamlit remembers the choice between runs.
        # name is the text of the chosen option, for example "DBSCAN".
        name = st.selectbox("Clustering algorithm", list(ALGORITHMS), key="single_algorithm")
        # look up the python file for that name. algorithm is now the module, for example dbscan_algo.
        algorithm = ALGORITHMS[name]
        # print the plain-language explanation that is written inside that algorithm's file.
        st.write(algorithm.DESCRIPTION)
        # ask the algorithm for an optional tip. for DBSCAN it says how far away the 10th nearest crime usually is.
        # other algorithms give back None (nothing).
        tip = algorithm.hint_text(hints)
        # if there is a tip (None counts as False), show it as small grey text.
        if tip:
            st.caption(tip)

        # a form means nothing runs until you press the button, however many parameters you change
        # normally, every time you move a slider the whole app re-runs. a form holds all the changes back
        # until the button inside the form is pressed. the form name has the algorithm in it,
        # so each algorithm gets its own form.
        with st.form(f"single_form_{name}"):
            # draw one widget for every parameter of the chosen algorithm (slider, number box, dropdown, tick box).
            # it gives back a dictionary of the chosen values, for example {"eps_m": 300.0, "min_samples": 10, ...}.
            # the middle part is a prefix that keeps widget names unique for each file and each algorithm.
            # hints is used to set the starting value of some parameters, like eps.
            params = render_params(algorithm.PARAMS, f"single_{ds.key}_{name}", hints)
            # the button that sends the form. it is True only in the run right after it was clicked, else False.
            # type="primary" makes it the coloured main button.
            run_clicked = st.form_submit_button("Run clustering", type="primary")

        # an expander is a box you can click to open or close.
        with st.expander("Map display settings"):
            # draw the map settings (heat type, radius, blur, tick boxes) and get their values as a dictionary.
            # these are outside the form on purpose, so changing them redraws the map without re-running the clustering.
            map_options = render_map_options("single")

    # start with "no error". we will fill this in if the clustering fails.
    error = None
    # only do the clustering in the run right after the button was clicked.
    if run_clicked:
        # "with right" = draw inside the right column. st.spinner shows a "Running ..." message while the work is going on.
        # we can put two "with" items on one line, separated by a comma.
        with right, st.spinner(f"Running {name}..."):
            # try something that might fail.
            try:
                # run the algorithm on the metre positions (ds.X) with the chosen parameters,
                # then calculate the scores, and save everything in session_state so it survives the next re-run.
                # ds.token is saved inside the result, so we can later check that it belongs to the same data.
                st.session_state[RESULT_KEY] = make_result(name, algorithm, ds.X, ds.token, params)
            # if the algorithm raised one of these errors (for example "k is bigger than the number of points"), catch it.
            # exc is the error. this stops the app from crashing and lets us show a friendly message.
            except (ValueError, RuntimeError, MemoryError) as exc:
                # delete the old result, so we do not show an old map next to a new error.
                # the None means "do not complain if there was nothing to delete".
                st.session_state.pop(RESULT_KEY, None)
                # keep the error text so we can show it below.
                error = str(exc)

    # everything indented under this is drawn in the right column.
    with right:
        # get the saved result from session_state. .get gives None if nothing is saved yet.
        result = st.session_state.get(RESULT_KEY)
        # case 1: the last run failed. show the error in a red box.
        if error:
            st.error(error)
        # case 2: we have a result AND it belongs to the data that is on screen now.
        # each result remembers a token (fingerprint of the data). if it matches ds.token the data has not changed.
        elif result and result["token"] == ds.token:
            # draw the scores, the heatmap, the hotspot table and the download buttons.
            # "single" is a prefix that keeps the widget names of this tab different from the compare tab.
            render_result(result, ds, map_options, "single")
        # case 3: nothing to show yet, or the result is old.
        else:
            # there is a saved result, but its token does not match. so the data (file, columns or filter) changed.
            if result:
                st.info("The data changed since the last run. Press Run clustering to update the results.")
            # there is no saved result at all, so the user has not run anything yet.
            else:
                st.info("Choose an algorithm, adjust its parameters, then press Run clustering. The heatmap and scores appear here.")