"""Run one algorithm on a dataset and score the result."""
from __future__ import annotations

import time
import uuid

import numpy as np

from .metrics import evaluate


def make_result(algorithm_name: str, algorithm, X: np.ndarray, token: str, params: dict) -> dict:
    """Cluster X with `algorithm`, score it, and return everything the UI needs."""
    start = time.perf_counter()
    labels = np.asarray(algorithm.run(X, **params), dtype=int)
    runtime = time.perf_counter() - start
    return {
        "id": uuid.uuid4().hex,
        "token": token,  # which dataset this belongs to
        "algorithm": algorithm_name,
        "params": dict(params),
        "labels": labels,
        "scores": evaluate(X, labels),
        "runtime_s": runtime,
    }
