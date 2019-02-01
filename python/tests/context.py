"""Provides easy import path procedure for test modules.

Font:
    "Structuring Your Project", The Hitchhiker's Guide to Python
    link: https://docs.python-guide.org/writing/structure/

Attributes:
    IMPORT_PATH_TUPLE (:obj:`tuple` of :obj:`str`): tuple containing extra
        import paths necessary to connect all test modules.

    TEST_PATH (:obj:`str`): Path for test datasets used in test procedures.

    DATASET_LIST (:obj:`list` of :obj:`pd.DataFrame`): Contains all test data
        used in test procedures.

    TARGET_COL_NAMES (:obj:`tuple` of :obj:`str`): tuple of column names of
        test datasets to be assumed as target columns.

    EPSILON (:obj:`float`): Very small value (<< 1) to use in asserts of floa-
        ting point correctness (in test cases) without numeric error problems.
"""
import typing as t
import os
import sys

import pandas as pd
import numpy as np
import rpy2.robjects
import rpy2.rinterface
import rpy2.robjects.packages
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri

IMPORT_PATH_TUPLE = ("..", "../pymfe")

for new_path in IMPORT_PATH_TUPLE:
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), new_path)))

import pymfe  # noqa: E402, F401
from pymfe.mfe import MFE  # noqa: E402

TARGET_COL_NAMES = ("target", )

EPSILON = 1.0e-6
EPSILON_RELAXED = 1.0e-2


def _get_test_data(path: str, file_extension: str = ".csv"
                   ) -> t.Sequence[pd.core.frame.DataFrame]:
    """Load test datasets (in alphabetical order).

    Args:
        path (:obj:`str`): path to directory where to look for test
            datasets.

        file_extension(:obj:`str`, optional): extension of test datasets
            files.

    Returns:
        list of pd.DataFrame: a list containing all dataframes loaded
            from given ``path``. The datasets are loaded in alphabetical
            order, so is the organization of returned list.
    """
    test_dataset_list = os.listdir(path)

    test_data = []
    for dataset_name in sorted(test_dataset_list):
        if dataset_name.endswith(file_extension):
            test_data.append(
                pd.read_csv(
                    "{0}{1}".format(path, dataset_name),
                    index_col=0,
                ))

    return test_data


TEST_PATH = "./test_datasets/"
DATASET_LIST = _get_test_data(TEST_PATH)


def get_col_indexes(dataset: pd.core.frame.DataFrame
                    ) -> t.Tuple[t.Sequence[bool], t.Sequence[bool]]:
    """Split columns of independent and target attributes.

    It is assumed that the target attribute of test datasets
    is in ``TARGET_COL_NAME``.

    Args:
        dataset (:obj:`pd.DataFrame`): a dataset in pandas DataFrame
            format.

    Returns:
        tuple[list, list]: two lists of boolean values marking, respec-
            tively, columns the are target features and columns that are
            independent attributes (the complement of the first list).
    """
    ind_targ = np.isin(dataset.columns, TARGET_COL_NAMES)
    ind_attr = ~ind_targ

    return ind_targ, ind_attr


def get_val_py(dataset: pd.core.frame.DataFrame, feat_name_py: str,
               ind_attr: t.Sequence[bool], ind_targ: t.Sequence[bool],
               method_args: t.Dict[str, t.Any], fit_args: t.Dict[str, t.Any],
               summary_name: str) -> t.Sequence:
    """Instantiate, Fit and Extract summarized metafeature from MFE.

    Args:
        dataset (:obj:`pd.DataFrame`): set of values for feature extraction.

        feat_name_py (:obj:`str`): name of the feature method in the Python
            MFE implementation to be extracted. Check ``MFE`` Class documen-
            tation for instructions of how to get a list of available metho-
            ds related with metafeature extraction in MFE Python implementa-
            tion.

        ind_attr (:obj:`Sequence` of :obj:`bool`): a sequence of booleans mar-
            king the columns of independent attributes of ``dataset``.

        ind_targ (:obj:`Sequence` of :obj:`bool`): a sequence of booleans mar-
            king the columns of target attributes of ``dataset``.

        method_args (:obj:`dict`): dictionary for extra arguments given for
            ``MFE.extract`` method. Check this method documentation for more
            in-depth details of this parameter format.

        fit_args (:obj:`dict`): dictionary for extra arguments given for
            ``MFE.fit`` method. Check this method documentation for more
            in-depth details of this parameter format.

        summary_name (:obj:`str`): name of the summary function to group the
            extracted feature values. Check ``MFE`` Class documentation for
            a list of available summary functions.

    Returns:
        sequence: the return values of ``MFE.Extract`` method, usually a se-
            quence containing one or more numeric values or ``np.nan`` obje-
            cts.

    Raises:
        Check documentation for ``MFE`` Class, ``MFE.fit`` and ``MFE.extract``
            for a list of possible expections. Exceptions related with incor-
            rect ``ind_attr`` and ``ind_targ`` arguments are also possible.
    """
    X = dataset.iloc[:, ind_attr].values
    y = dataset.iloc[:, ind_targ].values

    _, res_mfe_py_vals = MFE(
        features=feat_name_py,
        summary=summary_name,
    ).fit(
        X=X, y=y, **fit_args).extract(
            remove_nan=False, **method_args)

    return res_mfe_py_vals


def na_to_nan(values: t.Sequence[t.Any]) -> np.ndarray:
    """Convert any R ``NA`` object to :obj:`np.nan` within a list."""
    return np.array([
        np.nan if isinstance(x, rpy2.rinterface.NALogicalType) else x
        for x in list(values)
    ])


def get_val_r(dataset: pd.core.frame.DataFrame, feat_name_r: str,
              ind_attr: t.Sequence[bool], ind_targ: t.Sequence[bool],
              summary_name: str, group_name: str, **kwargs) -> t.Sequence:
    """Get summarized metafeature value from R mfe implementation.

    Args:
        dataset (:obj:`pd.DataFrame`): set of values for feature extraction.

        feat_name_r (:obj:`str`): name of the feature method in the R MFE
            implementation to be extracted. Check R ``mfe`` package documen-
            tation for orientation of how to get a list of available meta-
            features.

        ind_attr (:obj:`Sequence` of :obj:`bool`): a sequence of booleans mar-
            king the columns of independent attributes of ``dataset``.

        ind_targ (:obj:`Sequence` of :obj:`bool`): a sequence of booleans mar-
            king the columns of target attributes of ``dataset``.

        summary_name (:obj:`str`): name of the summary function to group the
            extracted feature values. Check R ``mfe`` package ``post.proces-
            sing`` object documentation for available summary functions.

        group_name (:obj:`str`): group name of R implementation where the
            desired feature are in. Available groups are {``general``,
            ``infotheo``, ``statistical``, ``landmarking``, ``model.based``}.

    Returns:
        sequence: the return values of ``MFE.Extract`` method, usually a se-
            quence containing one or more numeric values or ``np.nan`` obje-
            cts.

    Raises:
        AttributeError: if ``group_name`` is not a correct group name.
    """
    rpy2.robjects.pandas2ri.activate()
    rpy2.robjects.numpy2ri.activate()

    dataset_r = rpy2.robjects.pandas2ri.py2ri(dataset)

    mfe = rpy2.robjects.packages.importr("mfe")

    method_to_call = getattr(mfe, group_name)

    res_mfe_r = method_to_call(
        x=dataset_r.rx(ind_attr),
        y=dataset_r.rx(ind_targ),
        features=feat_name_r,
        summary=summary_name,
        **kwargs)

    rpy2.robjects.pandas2ri.deactivate()

    res_mfe_r_vals = na_to_nan(res_mfe_r.rx2(feat_name_r))

    return res_mfe_r_vals
