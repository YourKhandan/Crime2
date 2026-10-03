"""Tab 3: run several algorithms side by side and see which one scores best."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from algorithms import ALGORITHMS
from core.dataset import Dataset
from core.metrics import average_rank
from core.pipeline import make_result
from ui.result_view import DAVIES_BOULDIN_HELP, DUNN_HELP, SILHOUETTE_HELP, render_result
from ui.widgets import cached_hints, render_map_options, render_params

RESULTS_KEY = "compare_results"
HIGHLIGHT = "background-color: #cfe8ec; color: #0b3b44; font-weight: 600"


def _run_all(ds: Dataset, chosen: list[str], all_params: dict) -> dict:
    results, errors = {}, {}
    progress = st.progress(0.0, text="Starting...")
    for i, name in enumerate(chosen):
        progress.progress(i / len(chosen), text=f"Running {name}...")
        try:
            results[name] = make_result(name, ALGORITHMS[name], ds.X, ds.token, all_params[name])
        except (ValueError, RuntimeError, MemoryError) as exc:
            errors[name] = str(exc)
    progress.empty()
    return {"token": ds.token, "results": results, "errors": errors}


def _score_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        s = r["scores"]
        rows.append(
            {
                "Algorithm": name, "Clusters": s["n_clusters"], "Noise (%)": s["noise_pct"],
                "silhouette": s["silhouette"], "davies_bouldin": s["davies_bouldin"], "dunn": s["dunn"],
                "Time (s)": r["runtime_s"],
            }
        )
    table = pd.DataFrame(rows).set_index("Algorithm")
    table["Average rank (1 is best)"] = average_rank(table)
    return table


def render(ds: Dataset) -> None:
    hints = cached_hints(ds.key, ds.X)
    st.subheader("Compare algorithms")
    st.write(
        "Set each algorithm's parameters, then run them all on the same data. "
        "The table scores every algorithm with the same three measures so you can see which fits best."
    )

    with st.form("compare_form"):
        chosen = st.multiselect("Algorithms to compare", list(ALGORITHMS), default=list(ALGORITHMS), key="compare_choice")
        all_params: dict = {}
        grid = st.columns(2, gap="large")
        for i, (name, algorithm) in enumerate(ALGORITHMS.items()):
            with grid[i % 2].expander(f"{name} parameters"):
                st.caption(algorithm.DESCRIPTION)
                all_params[name] = render_params(algorithm.PARAMS, f"compare_{ds.key}_{name}", hints)
        run_clicked = st.form_submit_button("Compare algorithms", type="primary")

    if run_clicked:
        if not chosen:
            st.error("Choose at least one algorithm to compare.")
        else:
            st.session_state[RESULTS_KEY] = _run_all(ds, chosen, all_params)

    state = st.session_state.get(RESULTS_KEY)
    if not state:
        st.info("Press Compare algorithms to see the scores and heatmaps.")
        return
    if state["token"] != ds.token:
        st.info("The data changed since the last comparison. Press Compare algorithms to update the results.")
        return

    for name, message in state["errors"].items():
        st.error(f"{name} could not run: {message}")
    results = state["results"]
    if not results:
        return

    table = _score_table(results)
    best_rank = table["Average rank (1 is best)"].min()
    winners = list(table.index[table["Average rank (1 is best)"] == best_rank])

    st.markdown("#### Scores")
    shown = table.rename(
        columns={"silhouette": "Silhouette (higher)", "davies_bouldin": "Davies-Bouldin (lower)", "dunn": "Dunn (higher)"}
    )
    styled = (
        shown.style.format(
            {
                "Noise (%)": "{:.1f}", "Silhouette (higher)": "{:.3f}", "Davies-Bouldin (lower)": "{:.3f}",
                "Dunn (higher)": "{:.4f}", "Time (s)": "{:.2f}", "Average rank (1 is best)": "{:.2f}",
            },
            na_rep="n/a",
        )
        .highlight_max(subset=["Silhouette (higher)", "Dunn (higher)"], props=HIGHLIGHT)
        .highlight_min(subset=["Davies-Bouldin (lower)", "Average rank (1 is best)"], props=HIGHLIGHT)
    )
    st.dataframe(styled)

    if len(results) > 1:
        who = " and ".join(winners)
        st.success(f"Best on these three scores: {who} (average rank {best_rank:.2f}).")
    with st.expander("How to read these scores"):
        st.markdown(
            f"- **Silhouette**: {SILHOUETTE_HELP}\n- **Davies-Bouldin**: {DAVIES_BOULDIN_HELP}\n- **Dunn**: {DUNN_HELP}\n\n"
            "Two cautions. All three scores favour compact, round clusters, so K-Means often wins even when "
            "DBSCAN or Infomap trace the real hotspots better, for example a hotspot stretched along a street. "
            "And they ignore noise points, so an algorithm that leaves many points out can look better than it is. "
            "Check the Clusters and Noise columns and look at the heatmaps before you decide."
        )

    c1, c2, c3 = st.columns(3)
    c1.markdown("**Silhouette** (higher is better)")
    c1.bar_chart(table["silhouette"])
    c2.markdown("**Davies-Bouldin** (lower is better)")
    c2.bar_chart(table["davies_bouldin"])
    c3.markdown("**Dunn** (higher is better)")
    c3.bar_chart(table["dunn"])

    st.markdown("#### Heatmap")
    names = list(results)
    pick = st.selectbox("Show the heatmap for", names, index=names.index(winners[0]), key="compare_view")
    with st.expander("Map display settings"):
        map_options = render_map_options("compare")
    render_result(results[pick], ds, map_options, "compare")
