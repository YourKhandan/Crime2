"""KNN clustering (nearest-neighbour graph, Jarvis-Patrick style).

Plain KNN is a *supervised* method: it needs labelled examples. To use "K nearest neighbours" for
clustering we use the neighbour lists themselves:

1. every point lists its k nearest points;
2. two points are linked if they list each other (optional) and share enough neighbours;
3. each connected group of linked points is a cluster; tiny groups become noise.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

from .base import Param, relabel_by_size

NAME = "KNN"

DESCRIPTION = (
    "Every crime writes down its k closest neighbours. Two crimes become 'friends' when they are "
    "on each other's list and share enough common neighbours. Each group of friends is a hotspot, "
    "and crimes with too few friends are noise. (Plain KNN needs labelled data, so here the "
    "neighbour lists are used to group points instead.)"
)

PARAMS = [
    Param(
        "n_neighbors", "Neighbours per crime (k)", "int", default=15, min=3, max=100, step=1,
        help="How many of its closest crimes each crime looks at.",
    ),
    Param(
        "min_shared", "Shared neighbours needed", "int", default=3, min=0, max=50, step=1,
        help="How many neighbours two crimes must have in common to be linked. "
        "Higher means stricter, tighter hotspots. 0 switches the check off.",
    ),
    Param(
        "mutual", "Both crimes must list each other", "bool", default=True,
        help="Stricter, and it stops long chains of crimes from being joined into one giant cluster.",
    ),
    Param(
        "max_distance_m", "Ignore neighbours farther than (metres)", "float", default=300.0,
        min=0.0, max=50_000.0, step=10.0, hint="eps_m",
        help="Links longer than this are cut. Without a limit, neighbouring hotspots tend to chain "
        "together into one giant cluster. 0 means no limit.",
    ),
    Param(
        "min_cluster_size", "Smallest cluster to keep", "int", default=10, min=1, max=500, step=1,
        help="Groups smaller than this are treated as noise.",
    ),
]


def hint_text(hints: dict) -> str | None:
    return None


def _shared_neighbour_counts(A: sp.csr_matrix, rows: np.ndarray, cols: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """For each edge (i, j): how many neighbours do i and j have in common?"""
    out = np.empty(len(rows), dtype=np.int32)
    for start in range(0, len(rows), chunk):
        sl = slice(start, start + chunk)
        both = A[rows[sl]].multiply(A[cols[sl]])
        out[sl] = np.asarray(both.sum(axis=1)).ravel()
    return out


def run(
    X: np.ndarray,
    n_neighbors=15,
    min_shared=3,
    mutual=True,
    max_distance_m=300.0,
    min_cluster_size=10,
) -> np.ndarray:
    n = len(X)
    if n < 4:
        raise ValueError("KNN clustering needs at least 4 points.")
    k = int(min(n_neighbors, n - 1))

    # 1. neighbour lists (we ask for k+1 because a point is its own nearest neighbour)
    dist, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    rows = np.repeat(np.arange(n), k + 1)
    cols = idx.ravel()
    d = dist.ravel()
    keep = rows != cols
    if max_distance_m and max_distance_m > 0:
        keep &= d <= float(max_distance_m)
    rows, cols = rows[keep], cols[keep]

    # A[i, j] = 1 when j is on i's list
    A = sp.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(n, n))

    # 2. which pairs may be linked?
    pairs = A.multiply(A.T) if mutual else (A + A.T)
    pairs = sp.triu(pairs, k=1).tocoo()
    er, ec = pairs.row, pairs.col
    if min_shared and int(min_shared) > 0 and len(er):
        shared = _shared_neighbour_counts(A, er, ec)
        ok = shared >= int(min_shared)
        er, ec = er[ok], ec[ok]

    # 3. connected groups are the clusters
    graph = sp.coo_matrix((np.ones(len(er), dtype=np.int8), (er, ec)), shape=(n, n))
    _, groups = connected_components(graph, directed=False)
    return relabel_by_size(groups, min_cluster_size=int(min_cluster_size))
