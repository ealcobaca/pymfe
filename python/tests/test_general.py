"""Test module for General class metafeatures."""
import pytest

import context


class TestGeneral:
    """TestClass dedicated to test General metafeatures."""

    @pytest.mark.parametrize(
        ("dt_id, fit_args_py, feat_name_py,"
         "method_args_py, feat_name_r, summary_name, fit_args_r"),
        (
            (0, {}, "attr_to_inst", {}, "attrToInst", "all", {}),
            (1, {}, "attr_to_inst", {}, "attrToInst", "all", {}),
            (0, {}, "nr_inst", {}, "nrInst", "all", {}),
            (1, {}, "nr_inst", {}, "nrInst", "all", {}),
            (0, {}, "cat_to_num", {}, "catToNum", "all", {}),
            (1, {
                "check_bool": True
            }, "cat_to_num", {}, "catToNum", "all", {}),
            (0, {}, "nr_attr", {}, "nrAttr", "all", {}),
            (1, {}, "nr_attr", {}, "nrAttr", "all", {}),
            (0, {}, "nr_bin", {}, "nrBin", "all", {}),
            (1, {}, "nr_bin", {}, "nrBin", "all", {}),
            (0, {}, "cat_to_num", {}, "catToNum", "all", {}),
            (1, {
                "check_bool": True,
            }, "cat_to_num", {}, "catToNum", "all", {}),
            (0, {}, "freq_class", {}, "freqClass", "mean", {}),
            (1, {}, "freq_class", {}, "freqClass", "mean", {}),
            (0, {}, "freq_class", {}, "freqClass", "max", {}),
            (1, {}, "freq_class", {}, "freqClass", "max", {}),
            (0, {}, "freq_class", {}, "freqClass", "min", {}),
            (1, {}, "freq_class", {}, "freqClass", "min", {}),
            (0, {}, "freq_class", {}, "freqClass", "skewness", {}),
            (1, {}, "freq_class", {}, "freqClass", "skewness", {}),
            (0, {}, "freq_class", {}, "freqClass", "kurtosis", {}),
            (1, {}, "freq_class", {}, "freqClass", "kurtosis", {}),
            (0, {}, "freq_class", {}, "freqClass", "var", {}),
            (1, {}, "freq_class", {}, "freqClass", "var", {}),
            (0, {}, "freq_class", {}, "freqClass", "median", {}),
            (1, {}, "freq_class", {}, "freqClass", "median", {}),
            (0, {}, "freq_class", {}, "freqClass", "sd", {}),
            (1, {}, "freq_class", {}, "freqClass", "sd", {}),
            (0, {}, "freq_class", {}, "freqClass", "histogram", {}),
            (1, {}, "freq_class", {}, "freqClass", "histogram", {}),
            (0, {}, "inst_to_attr", {}, "instToAttr", "all", {}),
            (1, {}, "inst_to_attr", {}, "instToAttr", "all", {}),
            (0, {}, "nr_cat", {}, "nrCat", "all", {}),
            (1, {
                "check_bool": True,
            }, "nr_cat", {}, "nrCat", "all", {}),
            (0, {}, "nr_class", {}, "nrClass", "all", {}),
            (1, {}, "nr_class", {}, "nrClass", "all", {}),
            (0, {}, "nr_num", {}, "nrNum", "all", {}),
            (1, {
                "check_bool": True,
            }, "nr_num", {}, "nrNum", "all", {}),
            (0, {}, "num_to_cat", {}, "numToCat", "all", {}),
            (1, {
                "check_bool": True,
            }, "num_to_cat", {}, "numToCat", "all", {}),
        ),
    )
    def test_ft_methods_general(self, dt_id, fit_args_py, feat_name_py,
                                method_args_py, feat_name_r, summary_name,
                                fit_args_r):
        """Compare metafeat. values against correspondent from R mfe."""
        dataset = context.DATASET_LIST[dt_id]

        ind_targ, ind_attr = context.get_col_indexes(dataset)

        res_mfe_py = context.get_val_py(dataset, feat_name_py, ind_attr,
                                        ind_targ, method_args_py, fit_args_py,
                                        summary_name)

        res_mfe_r = context.get_val_r(dataset, feat_name_r, ind_attr, ind_targ,
                                      summary_name, "general", **fit_args_r)

        assert context.compare_results(res_mfe_py, res_mfe_r)
