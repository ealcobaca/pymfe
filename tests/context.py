"""Provides easy import path procedure for test modules.

Font:
    "Structuring Your Project", The Hitchhiker's Guide to Python
    link: https://docs.python-guide.org/writing/structure/

Attributes:
    TEST_PATH (str): Path for test datasets used in test procedures.
    DATASET_LIST (list): Contains all test data used in test procedures.
"""
import os
import pandas as pd

import sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mfe  # noqa: E402, F401


def _get_test_data(path):
    """Load test datasets."""
    test_dataset_list = os.listdir(path)
    print(test_dataset_list)

    test_data = []
    for dataset_name in test_dataset_list:
        test_data.append(
            pd.read_csv(
                path + dataset_name,
                index_col=0,
            ))

    return test_data


TEST_PATH = "./test_datasets/"
DATASET_LIST = _get_test_data(TEST_PATH)
