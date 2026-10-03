"""Cluster quality scores.

All three scores are computed on the clustered points only (noise, label -1, is left out) and on
x/y positions in metres.

    Silhouette      higher is better  (-1 to 1)
    Davies-Bouldin  lower is better   (0 and up)
    Dunn            higher is better  (0 and up)

On big datasets Silhouette and Dunn work on a sample, because they compare every point with every
other point. The sample keeps every cluster represented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from sklearn.metrics import davies_bouldin_score, silhouette_score

NAN = float("nan")


def _stratified_sample(labels: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    """Indices of a sample of about max_n points with at least 2 from every cluster."""
    if len(labels) <= max_n:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    fraction = max_n / len(labels)
    picked = []
    for cid in np.unique(labels):
        members = np.flatnonzero(labels == cid)
        take = min(len(members), max(2, int(round(len(members) * fraction))))
        picked.append(rng.choice(members, size=take, replace=False))
    return np.sort(np.concatenate(picked))


def dunn_index(X: np.ndarray, labels: np.ndarray, max_samples: int = 4000, seed: int = 42) -> float:
    """Smallest gap between two clusters divided by the widest cluster."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    ids = np.unique(labels)
    if len(ids) < 2:
        return NAN

    idx = _stratified_sample(labels, max_samples, seed)
    X, labels = X[idx], labels[idx]

    widest = 0.0
    for cid in ids:
        members = X[labels == cid]
        if len(members) > 1:
            widest = max(widest, float(pdist(members).max()))
    if widest == 0.0:
        return NAN

    smallest_gap = np.inf
    for cid in ids:
        inside = labels == cid
        if inside.all() or not (~inside).any():
            continue
        tree = cKDTree(X[inside])
        nearest, _ = tree.query(X[~inside], k=1, distance_upper_bound=smallest_gap)
        if len(nearest):
            smallest_gap = min(smallest_gap, float(nearest.min()))
    if not np.isfinite(smallest_gap):
        return NAN
    return smallest_gap / widest


def evaluate(X: np.ndarray, labels: np.ndarray, silhouette_sample: int = 5000, dunn_sample: int = 4000, seed: int = 42) -> dict:
    """Cluster count, noise share and the three quality scores."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    clustered = labels >= 0
    Xc, lc = X[clustered], labels[clustered]
    n_clusters = int(len(np.unique(lc)))

    scores = {
        "n_points": int(len(labels)),
        "n_clusters": n_clusters,
        "n_noise": int((~clustered).sum()),
        "noise_pct": float(100.0 * (~clustered).mean()) if len(labels) else 0.0,
        "silhouette": NAN,
        "davies_bouldin": NAN,
        "dunn": NAN,
    }
    # every score needs at least 2 clusters and more points than clusters
    if n_clusters < 2 or len(Xc) <= n_clusters:
        return scores

    try:
        idx = _stratified_sample(lc, silhouette_sample, seed)
        if len(np.unique(lc[idx])) >= 2:
            scores["silhouette"] = float(silhouette_score(Xc[idx], lc[idx]))
    except ValueError:
        pass
    try:
        scores["davies_bouldin"] = float(davies_bouldin_score(Xc, lc))
    except ValueError:
        pass
    scores["dunn"] = float(dunn_index(Xc, lc, max_samples=dunn_sample, seed=seed))
    return scores


def average_rank(table: pd.DataFrame) -> pd.Series:
    """Average rank of each algorithm across the three scores (1 is best).

    `table` needs columns silhouette, davies_bouldin and dunn. Missing scores rank last.
    """
    ranks = pd.DataFrame(
        {
            "silhouette": table["silhouette"].rank(ascending=False, method="min", na_option="bottom"),
            "dunn": table["dunn"].rank(ascending=False, method="min", na_option="bottom"),
            "davies_bouldin": table["davies_bouldin"].rank(ascending=True, method="min", na_option="bottom"),
        }
    )
    return ranks.mean(axis=1)
