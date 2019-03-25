""" Utils Module.

    Useful functions to help in the tests.
"""

import arff
import pandas as pd


DATA_ID = [
    "tests/test_datasets/mix_aids.arff"
]


def load_xy(dt_id):
    """Returns a dataset loaded from arff file.
    """
    data = arff.load(open(DATA_ID[dt_id], 'r'))
    df = pd.DataFrame(data['data'])
    y = df.iloc[:, -1]
    X = df.iloc[:, :df.shape[1]-1]

    return (X, y)
