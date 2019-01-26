"""Test module for InfoTheory class metafeatures."""
import pytest

import numpy as np

import context


class TestInfoTheory:
    """TestClass dedicated to test Information Theory metafeatures."""

    @pytest.mark.parametrize(
        ("dt_id, fit_args, feat_name_py,"
         "method_args, feat_name_r, summary_name"),
        (
            (2, {}, "class_ent", {}, "classEnt", "all"),
            (2, {}, "eq_num_attr", {}, "eqNumAttr", "all"),
            (2, {}, "ns_ratio", {}, "nsRatio", "all"),
            (2, {}, "attr_conc", {}, "attrConc", "mean"),
            (2, {}, "attr_ent", {}, "attrEnt", "mean"),
            (2, {}, "class_conc", {}, "classConc", "mean"),
            (2, {}, "joint_ent", {}, "jointEnt", "mean"),
            (2, {}, "mut_inf", {}, "mutInf", "mean"),
            (2, {}, "attr_conc", {}, "attrConc", "sd"),
            (2, {}, "attr_ent", {}, "attrEnt", "sd"),
            (2, {}, "class_conc", {}, "classConc", "sd"),
            (2, {}, "joint_ent", {}, "jointEnt", "sd"),
            (2, {}, "mut_inf", {}, "mutInf", "sd"),
            (2, {}, "attr_conc", {}, "attrConc", "histogram"),
            (2, {}, "attr_ent", {}, "attrEnt", "histogram"),
            (2, {}, "class_conc", {}, "classConc", "histogram"),
            (2, {}, "joint_ent", {}, "jointEnt", "histogram"),
            (2, {}, "mut_inf", {}, "mutInf", "histogram"),
            (2, {}, "attr_conc", {}, "attrConc", "range"),
            (2, {}, "attr_ent", {}, "attrEnt", "skewness"),
            (2, {}, "class_conc", {}, "classConc", "quantiles"),
            (2, {}, "joint_ent", {}, "jointEnt", "min"),
            (2, {}, "mut_inf", {}, "mutInf", "max"),
        ),
    )
    def test_ft_methods_info_theory(self, dt_id, fit_args, feat_name_py,
                                    method_args, feat_name_r, summary_name):
        """Compare metafeat. values against correspondent from R mfe."""
        dataset = context.DATASET_LIST[dt_id]

        ind_targ, ind_attr = context.get_col_indexes(dataset)

        res_mfe_py = context.get_val_py(dataset, feat_name_py, ind_attr,
                                        ind_targ, method_args, fit_args,
                                        summary_name)

        res_mfe_r = context.get_val_r(dataset, feat_name_r, ind_attr, ind_targ,
                                      summary_name, "infotheo")

        assert (all(np.isnan(res_mfe_py) == np.isnan(res_mfe_r))
                or all((res_mfe_py - res_mfe_r) < context.EPSILON))
