"""Test module for General class metafeatures."""
import pytest

import numpy as np
import rpy2.robjects
import rpy2.rinterface
import rpy2.robjects.packages
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri

import context
from pymfe.mfe import MFE


class TestGeneral:
    """TestClass dedicated to test General metafeatures."""

    EPSILON = 1.0e-6
    """Small value to assert floating point correctness w/o numeric errors."""

    @classmethod
    def _get_col_indexes(cls, dataset):
        """Split columns of independent and target attributes.

        It is assumed that the target attribute of test datasets
        is in ``context.TARGET_COL_NAME``.
        """
        ind_targ = np.isin(dataset.columns, context.TARGET_COL_NAMES)
        ind_attr = ~ind_targ

        return ind_targ, ind_attr

    @classmethod
    def _get_val_py(cls, dataset, feat_name_py, ind_attr, ind_targ,
                    method_args, fit_args, summary_name):
        """Instantiate, Fit and Extract summarized metafeature from MFE."""
        X = dataset.iloc[:, ind_attr].values
        y = dataset.iloc[:, ind_targ].values

        _, res_mfe_py_vals = MFE(
            features=feat_name_py,
            summary=summary_name,
        ).fit(
            X=X, y=y, **fit_args).extract(**method_args)

        return res_mfe_py_vals

    @classmethod
    def _na_to_nan(cls, values):
        """Convert any R ``NA`` object to :obj:`np.nan` within a list."""
        return np.array([
            np.nan if isinstance(x, rpy2.rinterface.NALogicalType) else x
            for x in list(values)
        ])

    @classmethod
    def _get_val_r(cls, dataset, feat_name_r, ind_attr, ind_targ,
                   summary_name):
        """Get summarized metafeature value from R mfe implementation."""
        rpy2.robjects.pandas2ri.activate()
        rpy2.robjects.numpy2ri.activate()

        dataset_r = rpy2.robjects.pandas2ri.py2ri(dataset)

        mfe = rpy2.robjects.packages.importr("mfe")

        res_mfe_r = mfe.general(
            x=dataset_r.rx(ind_attr),
            y=dataset_r.rx(ind_targ),
            features=feat_name_r,
            summary=summary_name)

        rpy2.robjects.pandas2ri.deactivate()

        res_mfe_r_vals = TestGeneral._na_to_nan(res_mfe_r.rx2(feat_name_r))

        return res_mfe_r_vals

    @pytest.mark.parametrize(
        ("dt_id, fit_args, feat_name_py,"
         "method_args, feat_name_r, summary_name"),
        (
            (0, {}, "attr_to_inst", {}, "attrToInst", "all"),
            (1, {}, "attr_to_inst", {}, "attrToInst", "all"),
            (0, {}, "nr_inst", {}, "nrInst", "all"),
            (1, {}, "nr_inst", {}, "nrInst", "all"),
            (0, {
                "check_bool": False
            }, "cat_to_num", {}, "catToNum", "all"),
            (1, {}, "cat_to_num", {}, "catToNum", "all"),
            (0, {}, "nr_attr", {}, "nrAttr", "all"),
            (1, {}, "nr_attr", {}, "nrAttr", "all"),
            (0, {}, "nr_bin", {}, "nrBin", "all"),
            (1, {}, "nr_bin", {}, "nrBin", "all"),
            (0, {
                "check_bool": False
            }, "cat_to_num", {}, "catToNum", "all"),
            (1, {}, "cat_to_num", {}, "catToNum", "all"),
            (0, {}, "freq_class", {}, "freqClass", "mean"),
            (1, {}, "freq_class", {}, "freqClass", "mean"),
            (0, {}, "freq_class", {}, "freqClass", "max"),
            (1, {}, "freq_class", {}, "freqClass", "max"),
            (0, {}, "freq_class", {}, "freqClass", "min"),
            (1, {}, "freq_class", {}, "freqClass", "min"),
            (0, {}, "freq_class", {}, "freqClass", "skewness"),
            (1, {}, "freq_class", {}, "freqClass", "skewness"),
            (0, {}, "freq_class", {}, "freqClass", "kurtosis"),
            (1, {}, "freq_class", {}, "freqClass", "kurtosis"),
            (0, {}, "freq_class", {}, "freqClass", "quantiles"),
            (1, {}, "freq_class", {}, "freqClass", "quantiles"),
            (0, {}, "freq_class", {}, "freqClass", "var"),
            (1, {}, "freq_class", {}, "freqClass", "var"),
            (0, {}, "freq_class", {}, "freqClass", "median"),
            (1, {}, "freq_class", {}, "freqClass", "median"),
            (0, {}, "freq_class", {
                "sd": {
                    "ddof": 1
                }
            }, "freqClass", "sd"),
            (1, {}, "freq_class", {
                "sd": {
                    "ddof": 1
                }
            }, "freqClass", "sd"),
            (0, {}, "freq_class", {
                "histogram": {
                    "bins": 10,
                    "normalize": True
                }
            }, "freqClass", "histogram"),
            (1, {}, "freq_class", {
                "histogram": {
                    "bins": 10,
                    "normalize": True
                }
            }, "freqClass", "histogram"),
            (0, {}, "inst_to_attr", {}, "instToAttr", "all"),
            (1, {}, "inst_to_attr", {}, "instToAttr", "all"),
            (0, {
                "check_bool": False
            }, "nr_cat", {}, "nrCat", "all"),
            (1, {}, "nr_cat", {}, "nrCat", "all"),
            (0, {}, "nr_class", {}, "nrClass", "all"),
            (1, {}, "nr_class", {}, "nrClass", "all"),
            (0, {
                "check_bool": False
            }, "nr_num", {}, "nrNum", "all"),
            (1, {}, "nr_num", {}, "nrNum", "all"),
            (0, {
                "check_bool": False
            }, "num_to_cat", {}, "numToCat", "all"),
            (1, {}, "num_to_cat", {}, "numToCat", "all"),
        ),
    )
    def test_ft_attr_to_inst(self, dt_id, fit_args, feat_name_py, method_args,
                             feat_name_r, summary_name):
        """Compare metafeat. values against correspondent from R mfe."""
        dataset = context.DATASET_LIST[dt_id]

        ind_targ, ind_attr = TestGeneral._get_col_indexes(dataset)

        res_mfe_py = TestGeneral._get_val_py(dataset, feat_name_py, ind_attr,
                                             ind_targ, method_args, fit_args,
                                             summary_name)

        res_mfe_r = TestGeneral._get_val_r(dataset, feat_name_r, ind_attr,
                                           ind_targ, summary_name)

        assert (all(np.isnan(res_mfe_py) == np.isnan(res_mfe_r))
                or all(res_mfe_py - res_mfe_r < TestGeneral.EPSILON))
