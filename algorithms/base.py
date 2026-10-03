"""Pieces shared by every clustering algorithm file.

Each algorithm file (kmeans_algo.py, dbscan_algo.py, ...) exposes the same four things:

    NAME         short name shown in the app
    DESCRIPTION  plain-language explanation shown above the parameters
    PARAMS       list of Param objects -> the app builds one widget per Param
    run(X, **params) -> labels   (X is an n x 2 array of x/y positions in metres)

Labels are integers: 0 is the biggest cluster, 1 the next, and so on.
-1 means "noise" (a point that belongs to no cluster).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class Param:
    """One hyper-parameter the user can change.

    The Streamlit UI turns every Param into a widget automatically, so adding a new
    hyper-parameter to an algorithm is one extra line in that algorithm's PARAMS list.
    """

    name: str  # keyword argument of run()
    label: str  # text above the widget
    kind: str  # "int" (slider) | "int_input" | "float" | "select" | "bool"
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: Sequence[Any] | None = None
    help: str = ""
    hint: str | None = None  # key in the dataset "hints" dict that overrides `default`


def relabel_by_size(labels: np.ndarray, min_cluster_size: int = 1) -> np.ndarray:
    """Renumber clusters 0, 1, 2 ... from biggest to smallest.

    Clusters with fewer than ``min_cluster_size`` points become noise (-1),
    and points that were already noise (-1) stay noise.
    """
    labels = np.asarray(labels)
    out = np.full(labels.shape, -1, dtype=int)
    valid = labels >= 0
    if not valid.any():
        return out

    ids, counts = np.unique(labels[valid], return_counts=True)
    big_enough = counts >= max(1, int(min_cluster_size))
    ids, counts = ids[big_enough], counts[big_enough]
    if len(ids) == 0:
        return out

    order = np.argsort(-counts, kind="stable")  # biggest first
    mapping = np.full(int(labels.max()) + 1, -1, dtype=int)
    mapping[ids[order]] = np.arange(len(order))
    out[valid] = mapping[labels[valid]]
    return out
