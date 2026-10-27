"""Pieces shared by every clustering algorithm file.

Each algorithm file (kmeans_algo.py, dbscan_algo.py, ...) exposes the same five things:

    NAME              short name shown in the app
    DESCRIPTION       plain-language explanation shown above the parameters
    PARAMS            list of Param objects -> the app builds one widget per Param
    hint_text(hints)  a tip shown under the description; it must exist, return None when there is no tip
    run(X, **params) -> labels   (X is an n x 2 array of x/y positions in metres)

Labels are integers: 0 is the biggest cluster, 1 the next, and so on.
-1 means "noise" (a point that belongs to no cluster).
"""
# ^ docstring: a note about this file. python does not run it, it is only for humans.
# this is the "rule book": every algorithm file must have the five things listed above,
# because the app's tabs call them without checking.

# lets us write type hints like "float | None" without errors on older python versions.
from __future__ import annotations

# dataclass is a shortcut for making a class that only holds data (like a form with fields).
from dataclasses import dataclass
# Any = "any kind of value is fine". Sequence = "a list, tuple or anything in a row". both are only type hints.
from typing import Any, Sequence

# numpy is for fast maths on arrays of numbers. "np" is the short name.
import numpy as np


# @dataclass writes the boring code for us (the __init__ that saves each field, and a nice printout).
# frozen=True means that after a Param is created it is read-only: nobody can change its values later by accident.
@dataclass(frozen=True)
# a Param is a "description card" for ONE setting that the user can change, for example eps or k.
# the app reads these cards and builds a slider, number box, dropdown or tick box for each one.
class Param:
    """One hyper-parameter the user can change.

    The Streamlit UI turns every Param into a widget automatically, so adding a new
    hyper-parameter to an algorithm is one extra line in that algorithm's PARAMS list.
    """

    # each line below is a field of the card, written as  fieldname: type.
    # fields without "= something" must always be given when you make a Param.
    # fields with "= something" are optional, and use that value if you do not give one.
    # (python needs the must-give fields first and the optional ones after them.)

    # the exact name of the matching input of the algorithm's run() function, for example "eps_m".
    name: str  # keyword argument of run()
    # the text shown above the widget, for example "Number of clusters (k)".
    label: str  # text above the widget
    # what kind of widget to draw: "int" = slider, "int_input" = whole-number box, "float" = decimal box,
    # "select" = dropdown, "bool" = tick box. if you write any other word, widgets.py raises an error.
    kind: str  # "int" (slider) | "int_input" | "float" | "select" | "bool"
    # the value the widget starts with. Any means it can be a number, text or True/False.
    default: Any
    # smallest allowed value. "float | None" means a number OR None (not used). only sliders and number boxes need it.
    min: float | None = None
    # biggest allowed value (same idea as min).
    max: float | None = None
    # how much the value jumps when you click + / - or move the slider one notch.
    step: float | None = None
    # the list of choices, only used when kind is "select". Sequence[Any] means a list of anything.
    options: Sequence[Any] | None = None
    # the small "?" text shown when you hover over the widget. it starts as an empty text "".
    help: str = ""
    # an optional link to a "hint" worked out from the data (for example "eps_m" for DBSCAN's walking distance).
    # if it is set, the app uses the hint as the starting value instead of default.
    hint: str | None = None  # key in the dataset "hints" dict that overrides `default`


# this function tidies up the cluster numbers that an algorithm gives back.
# it takes the raw labels (one number per crime) and a smallest allowed cluster size,
# and gives back clean labels: biggest cluster = 0, next = 1, ... and noise = -1.
#
# a small example to follow along. labels = [2, 2, 5, 5, 5, -1, 7] and min_cluster_size = 2.
# cluster 5 has 3 crimes, cluster 2 has 2 crimes, cluster 7 has 1 crime (too small), and one crime is already noise.
# the result is [1, 1, 0, 0, 0, -1, -1].
def relabel_by_size(labels: np.ndarray, min_cluster_size: int = 1) -> np.ndarray:
    """Renumber clusters 0, 1, 2 ... from biggest to smallest.

    Clusters with fewer than ``min_cluster_size`` points become noise (-1),
    and points that were already noise (-1) stay noise.
    """
    # make sure labels is a numpy array, even if a normal list was given.
    labels = np.asarray(labels)
    # start with an answer where every crime is noise (-1). same shape as labels. dtype=int means whole numbers.
    out = np.full(labels.shape, -1, dtype=int)
    # True for every crime that has a real cluster number (0 or more), False for noise (-1).
    # in the example: [T, T, T, T, T, F, T]
    valid = labels >= 0
    # if there is no real cluster at all, give back the all-noise answer straight away.
    if not valid.any():
        return out

    # look only at the real clusters. np.unique gives each cluster number once (ids) and how many crimes it has (counts).
    # in the example: ids = [2, 5, 7] and counts = [2, 3, 1]
    ids, counts = np.unique(labels[valid], return_counts=True)
    # True for clusters that are big enough. max(1, ...) makes sure the limit is never less than 1.
    # in the example: [T, T, F]  (cluster 7 has only 1 crime)
    big_enough = counts >= max(1, int(min_cluster_size))
    # keep only the big enough clusters. in the example: ids = [2, 5] and counts = [2, 3]
    ids, counts = ids[big_enough], counts[big_enough]
    # if every cluster was too small, give back the all-noise answer.
    if len(ids) == 0:
        return out

    # np.argsort gives the order that would sort the list. putting - in front of counts makes it biggest first.
    # kind="stable" means clusters of equal size keep their old order.
    # in the example: order = [1, 0]  (position 1, which is cluster 5, comes first)
    order = np.argsort(-counts, kind="stable")  # biggest first
    # make a lookup table: "old cluster number -> new cluster number".
    # it has one slot for every possible old number (0 up to the biggest one), and every slot starts as -1 (= noise).
    # in the example: 8 slots, all -1
    mapping = np.full(int(labels.max()) + 1, -1, dtype=int)
    # fill in the new numbers: ids[order] lists the old numbers biggest first, and np.arange(...) is 0, 1, 2, ...
    # so the biggest cluster gets 0, the next gets 1, and so on.
    # in the example: ids[order] = [5, 2], so mapping[5] = 0 and mapping[2] = 1. mapping[7] stays -1.
    mapping[ids[order]] = np.arange(len(order))
    # for every crime with a real cluster number, look up its new number in the table.
    # crimes that were noise are not touched, so they stay -1 in out.
    # in the example: [1, 1, 0, 0, 0, -1, -1]
    out[valid] = mapping[labels[valid]]
    # give the clean labels back.
    return out
