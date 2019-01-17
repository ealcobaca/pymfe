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

    SUMMARY_MFE_R = (
        "histogram",
        "kurtosis",
        "max",
        "mean",
        "median",
        "min",
        "quantiles",
        "sd",
        "skewness",
        "var",
    )
    """Summary functions supported by the R version of MFE."""

    @pytest.mark.parametrize(
        "dt_id, fit_args, feat_name_py, method_args, feat_name_r",
        (
            (0, {}, "attr_to_inst", {}, "attrToInst"),
            (1, {}, "attr_to_inst", {}, "attrToInst"),
            (0, {}, "nr_inst", {}, "nrInst"),
            (1, {}, "nr_inst", {}, "nrInst"),
            (0, {"check_bool": False}, "cat_to_num", {}, "catToNum"),
            (1, {}, "cat_to_num", {}, "catToNum"),
            (0, {}, "nr_attr", {}, "nrAttr"),
            (1, {}, "nr_attr", {}, "nrAttr"),
            (0, {}, "nr_bin", {}, "nrBin"),
            (1, {}, "nr_bin", {}, "nrBin"),
            (0, {"check_bool": False}, "cat_to_num", {}, "catToNum"),
            (1, {}, "cat_to_num", {}, "catToNum"),
            (0, {}, "freq_class", {}, "freqClass"),
            (1, {}, "freq_class", {}, "freqClass"),
            (0, {}, "inst_to_attr", {}, "instToAttr"),
            (1, {}, "inst_to_attr", {}, "instToAttr"),
            (0, {"check_bool": False}, "nr_cat", {}, "nrCat"),
            (1, {}, "nr_cat", {}, "nrCat"),
            (0, {}, "nr_class", {}, "nrClass"),
            (1, {}, "nr_class", {}, "nrClass"),
            (0, {"check_bool": False}, "nr_num", {}, "nrNum"),
            (1, {}, "nr_num", {}, "nrNum"),
            (0, {"check_bool": False}, "num_to_cat", {}, "numToCat"),
            (1, {}, "num_to_cat", {}, "numToCat"),
        ),
    )
    def test_ft_attr_to_inst(
            self,
            dt_id,
            fit_args,
            feat_name_py,
            method_args,
            feat_name_r):
        """Test method ``."""
        dataset = context.DATASET_LIST[dt_id]

        ind_targ = dataset.columns == "target"
        ind_attr = ~ind_targ

        X = dataset.iloc[:, ind_attr].values
        y = dataset.iloc[:, ind_targ].values

        ext_names, ext_values = MFE(
            features=feat_name_py,
            summary=TestGeneral.SUMMARY_MFE_R,
            ).fit(X=X, y=y, **fit_args).extract(**method_args)

        mfe = rpy2.robjects.packages.importr("mfe")

        rpy2.robjects.pandas2ri.activate()

        dataset_r = rpy2.robjects.pandas2ri.py2ri(dataset)

        expected_val = mfe.general(
            x=dataset_r.rx(ind_attr),
            y=dataset_r.rx(ind_targ),
            features=feat_name_r)

        rpy2.robjects.pandas2ri.deactivate()

        expected_val = expected_val.rx2(feat_name_r)[0]

        if isinstance(expected_val, rpy2.rinterface.NALogicalType):
            expected_val = np.nan

        assert (ext_values[0] == expected_val
                or all(np.isnan((ext_values[0], expected_val))))
