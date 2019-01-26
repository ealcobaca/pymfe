"""Test module for General class metafeatures."""
import pytest

import numpy as np

import context


class TestGeneral:
    """TestClass dedicated to test General metafeatures."""

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
            (0, {}, "freq_class", {}, "freqClass", "sd"),
            (1, {}, "freq_class", {}, "freqClass", "sd"),
            (0, {}, "freq_class", {}, "freqClass", "histogram"),
            (1, {}, "freq_class", {}, "freqClass", "histogram"),
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
    def test_ft_methods_general(self, dt_id, fit_args, feat_name_py,
                                method_args, feat_name_r, summary_name):
        """Compare metafeat. values against correspondent from R mfe."""
        dataset = context.DATASET_LIST[dt_id]

        ind_targ, ind_attr = context.get_col_indexes(dataset)

        res_mfe_py = context.get_val_py(dataset, feat_name_py, ind_attr,
                                        ind_targ, method_args, fit_args,
                                        summary_name)

        res_mfe_r = context.get_val_r(dataset, feat_name_r, ind_attr, ind_targ,
                                      summary_name, "general")

        assert (all(np.isnan(res_mfe_py) == np.isnan(res_mfe_r))
                or all(res_mfe_py - res_mfe_r < context.EPSILON))
