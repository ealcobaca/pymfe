"""Provides easy import path procedure for test modules.

Font:
    "Structuring Your Project", The Hitchhiker's Guide to Python
    link: https://docs.python-guide.org/writing/structure/

Attributes:
    TEST_PATH (:obj:`str`): Path for test datasets used in test procedures.

    DATASET_LIST (:obj:`list`): Contains all test data used in test procedures.

    TARGET_COL_NAMES (:obj:`tuple` of :obj:`str`): tuple of column names of
        test datasets to be assumed as target columns.
"""
import os
import sys

import pandas as pd

IMPORT_PATH_TUPLE = ("..", "../pymfe", "../../rwrapper/pymfe/mfe/")

for new_path in IMPORT_PATH_TUPLE:
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), new_path)))

import pymfe  # noqa: E402, F401

TARGET_COL_NAMES = ("target", )


def _get_test_data(path):
    """Load test datasets."""
    test_dataset_list = os.listdir(path)

    test_data = []
    for dataset_name in test_dataset_list:
        if dataset_name.endswith(".csv"):
            test_data.append(pd.read_csv(
                path + dataset_name,
                index_col=0,
            ))

    return test_data


TEST_PATH = "./test_datasets/"
DATASET_LIST = _get_test_data(TEST_PATH)
