"""Louvain clustering.

Louvain finds communities in a *network*, so (like Infomap) we first turn the crime locations into one:
every crime is linked to its k nearest crimes, and closer links are stronger.
"""
# ^ docstring: a short note about this file. python does not run it, it is only for humans.

# lets us write type hints like "str | None" without errors on older python versions.
from __future__ import annotations

# numpy is for fast maths on arrays of numbers. "np" is the short name.
import numpy as np

# louvain_communities is the ready-made Louvain function from the networkx library.
# we import it from its full path so it also works on older networkx versions.
from networkx.algorithms.community import louvain_communities
# Graph is networkx's "network" object: a bunch of dots (nodes) joined by lines (edges).
from networkx import Graph

# Param describes one setting the user can change. relabel_by_size renumbers clusters biggest first
# and turns tiny clusters into noise (-1). both live in algorithms/base.py.
from .base import Param, relabel_by_size
# we re-use the function that builds the "nearest neighbours network" from infomap_algo.py,
# so we do not have to write the same code twice. Louvain and Infomap both work on this same kind of network.
from .infomap_algo import WEIGHT_OPTIONS, _build_edges

# the name shown in the dropdown and in the tables. it is also the key in the ALGORITHMS dictionary.
NAME = "Louvain"

# a plain-language explanation. the app shows it above the parameters.
DESCRIPTION = (
    "Imagine every crime is a person, and nearby crimes are friends. Louvain lets each person join "
    "the friend group that suits them best, then treats each group as one big person and repeats. "
    "It stops when no one wants to move. You don't choose the number of clusters."
)

# the list of settings the user can change. the app draws one widget for each Param automatically.
PARAMS = [
    Param(
        # name = the name of the matching input of run() below. label = the text shown above the widget.
        # kind "int" = a slider that gives whole numbers. default/min/max/step = start, lowest, highest, jump size.
        "n_neighbors", "Neighbours per crime (k)", "int", default=20, min=3, max=60, step=1,
        # help = the small "?" text shown when you hover over the widget.
        help="How many of its closest crimes each crime is linked to when the network is built.",
    ),
    Param(
        # kind "select" = a dropdown. options = the choices in it. we re-use the same three choices as Infomap.
        "weight", "Link strength", "select", default="Gaussian", options=WEIGHT_OPTIONS,
        help="Gaussian: close links count much more than far ones. Inverse distance: a gentler "
        "fall-off. Equal: every link counts the same.",
    ),
    Param(
        # kind "float" = a number box that allows decimals.
        "resolution", "Resolution", "float", default=0.2, min=0.05, max=5.0, step=0.05,
        help="Zoom dial. Higher gives more, smaller hotspots; lower gives fewer, bigger ones.",
    ),
    Param(
        "min_cluster_size", "Smallest cluster to keep", "int", default=20, min=1, max=500, step=1,
        help="Groups smaller than this are treated as noise.",
    ),
    Param(
        # kind "int_input" = a number box for whole numbers (used for the random seed).
        "seed", "Random seed", "int_input", default=42, min=1, max=1_000_000, step=1,
        help="Change it to get a different, but repeatable, result.",
    ),
]


# every algorithm file must have this function, because the cluster tab calls algorithm.hint_text(hints)
# for whichever algorithm is selected. Louvain has no special tip, so it gives back None (nothing).
def hint_text(hints: dict) -> str | None:
    return None


# this does the real work. X is a table of x/y positions in metres, one row per crime.
# the other inputs are the settings from PARAMS above, and the values after "=" are used if none are given.
# it gives back one number per crime: its cluster number (0 = biggest cluster, -1 = noise).
def run(
    X: np.ndarray,
    n_neighbors=20,
    weight="Gaussian",
    resolution=0.2,
    min_cluster_size=20,
    seed=42,
) -> np.ndarray:
    # n = how many crimes (rows) we have.
    n = len(X)
    # a network with fewer than 4 dots is meaningless, so stop with a clear message.
    if n < 4:
        raise ValueError("Louvain needs at least 4 points.")
    # k can never be bigger than "number of other crimes" (n - 1), so take the smaller of the two.
    k = int(min(n_neighbors, n - 1))
    # build the network. a and b are lists of crime numbers: crime a[i] is linked to crime b[i].
    # w holds the strength of each link (close = strong). this function is written in infomap_algo.py.
    a, b, w = _build_edges(X, k, weight)

    # make an empty network.
    graph = Graph()
    # add one dot for every crime, numbered 0 to n-1. we do this first so that a crime with no links still exists.
    graph.add_nodes_from(range(n))
    # add all the links. zip(a, b, w) walks through the three lists together: (crime, crime, strength).
    # .tolist() changes numpy numbers into normal python numbers, which networkx prefers.
    graph.add_weighted_edges_from(zip(a.tolist(), b.tolist(), w.tolist()))

    # run Louvain. it gives back a list of groups, and every group is a set of crime numbers.
    # weight="weight" means "use the link strengths we added". resolution is the zoom dial from above.
    # seed makes the result repeatable (Louvain uses some randomness).
    groups = louvain_communities(graph, weight="weight", resolution=float(resolution), seed=int(seed))

    # start by marking every crime as noise (-1).
    labels = np.full(n, -1, dtype=int)
    # enumerate gives a counter (group_id = 0, 1, 2, ...) together with each group.
    for group_id, members in enumerate(groups):
        # give all the crimes in this group the same cluster number.
        labels[list(members)] = group_id
    # renumber the clusters biggest first, and turn clusters smaller than min_cluster_size into noise.
    return relabel_by_size(labels, min_cluster_size=int(min_cluster_size))