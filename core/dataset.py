"""The cleaned data that the clustering tabs work on."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    df: pd.DataFrame  # columns: lat, lon, offence, file_row (valid rows only, after filters)
    X: np.ndarray  # same points as x/y positions in metres
    token: str  # changes whenever the data under analysis changes
    key: str  # stable for one file + column choice (used for widget keys)
    source_name: str
    n_valid_total: int  # valid rows before the offence filter and the point cap
    sampled: bool  # True if a random sample was taken because of the point cap
