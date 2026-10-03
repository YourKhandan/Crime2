"""Dataset hints and per-cluster summary tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def compute_hints(X: np.ndarray, k: int = 10, seed: int = 0) -> dict:
    """Data-driven starting values for parameters (for example DBSCAN's eps)."""
    hints = {"eps_m": 300.0, "kdist_median_m": float("nan")}
    n = len(X)
    if n <= k + 1:
        return hints
    tree = NearestNeighbors(n_neighbors=k + 1).fit(X)
    rng = np.random.default_rng(seed)
    probe = X if n <= 20_000 else X[rng.choice(n, size=20_000, replace=False)]
    dist, _ = tree.kneighbors(probe)
    kth = dist[:, -1]  # distance to the k-th nearest other crime
    hints["kdist_median_m"] = float(np.median(kth))
    hints["eps_m"] = float(max(10.0, round(np.percentile(kth, 70) / 10.0) * 10.0))
    return hints


SUMMARY_COLUMNS = ["cluster", "incidents", "share_pct", "centre_lat", "centre_lon", "top_offences"]


def summarise_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """One row per cluster, biggest first. Cluster numbers start at 1; noise is left out."""
    labels = np.asarray(labels)
    clustered = labels >= 0
    if not clustered.any():
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    d = pd.DataFrame(
        {
            "cluster": labels[clustered] + 1,
            "lat": df["lat"].to_numpy()[clustered],
            "lon": df["lon"].to_numpy()[clustered],
            "offence": df["offence"].to_numpy()[clustered],
        }
    )
    out = d.groupby("cluster").agg(incidents=("lat", "size"), centre_lat=("lat", "mean"), centre_lon=("lon", "mean")).reset_index()
    out["share_pct"] = 100.0 * out["incidents"] / len(df)

    counts = d.groupby(["cluster", "offence"]).size().rename("n").reset_index()
    counts = counts.merge(out[["cluster", "incidents"]], on="cluster")
    counts = counts.sort_values(["cluster", "n"], ascending=[True, False])
    counts["rank"] = counts.groupby("cluster").cumcount()
    top = counts[counts["rank"] < 3].copy()
    top["text"] = top["offence"].astype(str) + " " + (100.0 * top["n"] / top["incidents"]).round().astype(int).astype(str) + "%"
    out["top_offences"] = out["cluster"].map(top.groupby("cluster")["text"].agg(", ".join))

    out = out.sort_values("incidents", ascending=False).reset_index(drop=True)
    return out[SUMMARY_COLUMNS]
