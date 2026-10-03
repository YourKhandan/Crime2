"""Infomap clustering.

Infomap finds communities in a *network*, so we first turn the crime locations into one:
every crime is linked to its k nearest crimes, and closer links are stronger.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import Param, relabel_by_size

NAME = "Infomap"

DESCRIPTION = (
    "Imagine someone wandering randomly from crime to crime along nearest-neighbour links. "
    "Places where the wanderer keeps circling before escaping are hotspots. Infomap finds these "
    "traps by itself, so you don't choose the number of clusters."
)

WEIGHT_OPTIONS = ["Gaussian", "Inverse distance", "Equal"]

PARAMS = [
    Param(
        "n_neighbors", "Neighbours per crime (k)", "int", default=30, min=3, max=60, step=1,
        help="How many of its closest crimes each crime is linked to when the network is built.",
    ),
    Param(
        "weight", "Link strength", "select", default="Gaussian", options=WEIGHT_OPTIONS,
        help="Gaussian: close links count much more than far ones. Inverse distance: a gentler "
        "fall-off. Equal: every link counts the same.",
    ),
    Param(
        "markov_time", "Wanderer's patience (Markov time)", "float", default=4.0, min=0.1, max=10.0, step=0.1,
        help="Zoom dial. Higher gives fewer, bigger hotspots; lower gives more, smaller ones. "
        "Nearest-neighbour networks tend to split into many tiny pieces at 1, so the default is higher.",
    ),
    Param(
        "num_trials", "Attempts", "int", default=5, min=1, max=20, step=1,
        help="Infomap tries this many times and keeps the best partition.",
    ),
    Param(
        "min_cluster_size", "Smallest cluster to keep", "int", default=10, min=1, max=500, step=1,
        help="Groups smaller than this are treated as noise.",
    ),
    Param(
        "seed", "Random seed", "int_input", default=42, min=1, max=1_000_000, step=1,
        help="Change it to get a different, but repeatable, result.",
    ),
]


def hint_text(hints: dict) -> str | None:
    return None


def _build_edges(X: np.ndarray, k: int, weight: str):
    """Undirected k-nearest-neighbour edges (i < j) with a strength for each."""
    n = len(X)
    dist, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    rows = np.repeat(np.arange(n), k + 1)
    cols = idx.ravel()
    d = dist.ravel()
    keep = rows != cols
    rows, cols, d = rows[keep], cols[keep], d[keep]

    # i->j and j->i describe the same link, so keep each pair once
    a, b = np.minimum(rows, cols), np.maximum(rows, cols)
    _, first = np.unique(a.astype(np.int64) * n + b, return_index=True)
    a, b, d = a[first], b[first], d[first]

    if weight == "Gaussian":
        positive = d[d > 0]
        sigma = float(np.median(positive)) if len(positive) else 1.0
        w = np.exp(-0.5 * (d / sigma) ** 2)
    elif weight == "Inverse distance":
        positive = d[d > 0]
        scale = float(np.median(positive)) if len(positive) else 1.0
        w = 1.0 / (1.0 + d / scale)
    else:
        w = np.ones_like(d)
    return a, b, np.maximum(w, 1e-9)


def run(
    X: np.ndarray,
    n_neighbors=30,
    weight="Gaussian",
    markov_time=4.0,
    num_trials=5,
    min_cluster_size=10,
    seed=42,
) -> np.ndarray:
    try:
        from infomap import Infomap
    except ImportError as exc:  # pragma: no cover - depends on the user's machine
        raise RuntimeError("The 'infomap' package is not installed. Run:  pip install infomap") from exc

    n = len(X)
    if n < 4:
        raise ValueError("Infomap needs at least 4 points.")
    k = int(min(n_neighbors, n - 1))
    a, b, w = _build_edges(X, k, weight)

    im = Infomap(f"--two-level --silent --num-trials {int(num_trials)} --seed {int(seed)} --markov-time {float(markov_time)}")
    for i, j, strength in zip(a.tolist(), b.tolist(), w.tolist()):
        im.add_link(i, j, strength)
    im.run()

    labels = np.full(n, -1, dtype=int)
    for node_id, module_id in im.get_modules().items():
        labels[node_id] = module_id
    return relabel_by_size(labels, min_cluster_size=int(min_cluster_size))
