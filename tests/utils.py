"""Utils Module.

Useful functions to help in the tests.
"""
import typing as t

import arff
import pandas as pd
import numpy as np

DATA_ID = [
    "tests/test_datasets/mix_aids.arff",
    "tests/test_datasets/cat_kr-vs-kp.arff",
    "tests/test_datasets/num_Iris.arff",
]

DATA_ = [
    None,
    None,
    None,
]


def load_xy(dt_id: int):
    """Returns a dataset loaded from arff file."""
    if DATA_[dt_id] is None:
        with open(DATA_ID[dt_id], "r") as data_file:
            data = arff.load(data_file)
            df = pd.DataFrame(data["data"])
            y = df.iloc[:, -1]
            X = df.iloc[:, :-1]
            DATA_[dt_id] = (X, y)

    return DATA_[dt_id]


def raise_memory_error(size: t.Union[int, float] = 1e20) -> np.ndarray:
    """Try to create a huge array, raising a MemoryError."""
    return np.zeros(int(size), dtype=np.float64)
