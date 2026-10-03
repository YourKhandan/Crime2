"""K-Means clustering."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from .base import Param, relabel_by_size

NAME = "K-Means"

DESCRIPTION = (
    "Drop k pins on the map. Every crime joins its nearest pin, then each pin slides to the "
    "middle of its crowd. Repeat until nothing moves. You choose k, and every crime ends up "
    "in some group, so there is no noise."
)

PARAMS = [
    Param(
        "n_clusters", "Number of clusters (k)", "int", default=8, min=2, max=50, step=1,
        help="How many pins to drop, i.e. how many hotspots to find.",
    ),
    Param(
        "init", "Starting pins", "select", default="k-means++",
        options=["k-means++", "random"],
        help="How the pins are first placed. k-means++ spreads them out; random drops them anywhere.",
    ),
    Param(
        "n_init", "Restarts", "int", default=10, min=1, max=30, step=1,
        help="How many times to start over with new pins. The best result is kept.",
    ),
    Param(
        "max_iter", "Max rounds per restart", "int", default=300, min=50, max=1000, step=50,
        help="Upper limit on slide-and-reassign rounds.",
    ),
    Param(
        "random_state", "Random seed", "int_input", default=42, min=0, max=1_000_000, step=1,
        help="Change it to get a different, but repeatable, result.",
    ),
]


def hint_text(hints: dict) -> str | None:
    return None


def run(X: np.ndarray, n_clusters=8, init="k-means++", n_init=10, max_iter=300, random_state=42) -> np.ndarray:
    n_clusters = int(n_clusters)
    if n_clusters >= len(X):
        raise ValueError(
            f"K-Means needs more points than clusters: {len(X)} points, k = {n_clusters}. "
            "Lower k or load more data."
        )
    model = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=int(n_init),
        max_iter=int(max_iter),
        random_state=int(random_state),
    )
    return relabel_by_size(model.fit_predict(X))
