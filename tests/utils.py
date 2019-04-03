""" Utils Module.

    Useful functions to help in the tests.
"""

import arff
import pandas as pd


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


def load_xy(dt_id):
    """Returns a dataset loaded from arff file.
    """
    if DATA_[dt_id] is None:
        with open(DATA_ID[dt_id], 'r') as data_file:
            data = arff.load(data_file)
            df = pd.DataFrame(data['data'])
            y = df.iloc[:, -1]
            X = df.iloc[:, :df.shape[1]-1]
            DATA_[dt_id] = (X, y)

    return DATA_[dt_id]
