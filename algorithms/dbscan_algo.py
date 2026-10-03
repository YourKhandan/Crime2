"""DBSCAN (density-based) clustering."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from .base import Param, relabel_by_size

NAME = "DBSCAN"

DESCRIPTION = (
    "A crime belongs to a hotspot if enough other crimes (min samples) sit within walking "
    "distance of it (eps). Crowds that touch merge into one hotspot, and lonely crimes are "
    "marked as noise. You don't choose the number of clusters."
)

PARAMS = [
    Param(
        "eps_m", "Walking distance, eps (metres)", "float", default=300.0, min=5.0, max=50_000.0, step=10.0,
        hint="eps_m",
        help="How close two crimes must be to count as neighbours. Bigger values merge hotspots; "
        "smaller values split them.",
    ),
    Param(
        "min_samples", "Crimes needed to form a crowd", "int", default=10, min=2, max=200, step=1,
        help="A spot is only 'crowded' if at least this many crimes are within eps.",
    ),
    Param(
        "metric", "Distance measure", "select", default="euclidean",
        options=["euclidean", "manhattan", "chebyshev"],
        help="euclidean is the straight line, manhattan counts city blocks, "
        "chebyshev uses the larger of the east-west and north-south gaps.",
    ),
]


def hint_text(hints: dict) -> str | None:
    value = hints.get("kdist_median_m")
    if value is None or not np.isfinite(value):
        return None
    return (
        f"In this data the 10th-nearest crime is usually about {value:,.0f} m away, "
        f"so eps of {hints.get('eps_m', value):,.0f} m is a sensible starting point."
    )


def _check_eps_is_safe(X: np.ndarray, eps: float, metric: str) -> None:
    """DBSCAN stores every point's neighbour list. A huge eps can eat all memory, so stop early."""
    rng = np.random.default_rng(0)
    probe = X[rng.choice(len(X), size=min(300, len(X)), replace=False)]
    tree = NearestNeighbors(radius=eps, metric=metric).fit(X)
    counts = [len(n) for n in tree.radius_neighbors(probe, return_distance=False)]
    estimated_links = float(np.mean(counts)) * len(X)
    if estimated_links > 200_000_000:
        raise ValueError(
            f"eps = {eps:,.0f} m puts about {np.mean(counts):,.0f} crimes inside every circle, which is too "
            "many to cluster. Lower eps, or reduce 'Maximum points to cluster' in the Data tab."
        )


def run(X: np.ndarray, eps_m=300.0, min_samples=10, metric="euclidean") -> np.ndarray:
    eps_m = float(eps_m)
    _check_eps_is_safe(X, eps_m, metric)
    labels = DBSCAN(eps=eps_m, min_samples=int(min_samples), metric=metric, n_jobs=-1).fit_predict(X)
    return relabel_by_size(labels)
