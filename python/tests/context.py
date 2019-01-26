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
import numpy as np
import rpy2.robjects
import rpy2.rinterface
import rpy2.robjects.packages
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri

IMPORT_PATH_TUPLE = ("..", "../pymfe", "../../rwrapper/pymfe/mfe/")

for new_path in IMPORT_PATH_TUPLE:
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), new_path)))

import pymfe  # noqa: E402, F401
from pymfe.mfe import MFE  # noqa: E402

TARGET_COL_NAMES = ("target", )

EPSILON = 1.0e-6
"""Small value to assert floating point correctness w/o numeric errors."""


def _get_test_data(path):
    """Load test datasets."""
    test_dataset_list = os.listdir(path)

    test_data = []
    for dataset_name in sorted(test_dataset_list):
        if dataset_name.endswith(".csv"):
            test_data.append(pd.read_csv(
                path + dataset_name,
                index_col=0,
            ))

    return test_data


TEST_PATH = "./test_datasets/"
DATASET_LIST = _get_test_data(TEST_PATH)


def get_col_indexes(dataset):
    """Split columns of independent and target attributes.

    It is assumed that the target attribute of test datasets
    is in ``TARGET_COL_NAME``.
    """
    ind_targ = np.isin(dataset.columns, TARGET_COL_NAMES)
    ind_attr = ~ind_targ

    return ind_targ, ind_attr


def get_val_py(dataset, feat_name_py, ind_attr, ind_targ, method_args,
               fit_args, summary_name):
    """Instantiate, Fit and Extract summarized metafeature from MFE."""
    X = dataset.iloc[:, ind_attr].values
    y = dataset.iloc[:, ind_targ].values

    _, res_mfe_py_vals = MFE(
        features=feat_name_py,
        summary=summary_name,
    ).fit(
        X=X, y=y, **fit_args).extract(**method_args)

    return res_mfe_py_vals


def na_to_nan(values):
    """Convert any R ``NA`` object to :obj:`np.nan` within a list."""
    return np.array([
        np.nan if isinstance(x, rpy2.rinterface.NALogicalType) else x
        for x in list(values)
    ])


def get_val_r(dataset, feat_name_r, ind_attr, ind_targ, summary_name,
              group_name):
    """Get summarized metafeature value from R mfe implementation."""
    rpy2.robjects.pandas2ri.activate()
    rpy2.robjects.numpy2ri.activate()

    dataset_r = rpy2.robjects.pandas2ri.py2ri(dataset)

    mfe = rpy2.robjects.packages.importr("mfe")

    method_to_call = getattr(mfe, group_name)

    res_mfe_r = method_to_call(
        x=dataset_r.rx(ind_attr),
        y=dataset_r.rx(ind_targ),
        features=feat_name_r,
        summary=summary_name)

    rpy2.robjects.pandas2ri.deactivate()

    res_mfe_r_vals = na_to_nan(res_mfe_r.rx2(feat_name_r))

    return res_mfe_r_vals
