"""Test module for InfoTheory class metafeatures."""
import pytest

import context


class TestInfoTheory:
    """TestClass dedicated to test Information Theory metafeatures."""

    @pytest.mark.parametrize(
        ("dt_id, fit_args_py, feat_name_py,"
         "method_args_py, feat_name_r, summary_name, fit_args_r"),
        (
            (2, {}, "class_ent", {}, "classEnt", "all", {}),
            (2, {}, "eq_num_attr", {}, "eqNumAttr", "all", {}),
            (2, {}, "ns_ratio", {}, "nsRatio", "all", {}),
            (2, {}, "attr_conc", {}, "attrConc", "mean", {}),
            (2, {}, "attr_ent", {}, "attrEnt", "mean", {}),
            (2, {}, "class_conc", {}, "classConc", "mean", {}),
            (2, {}, "joint_ent", {}, "jointEnt", "mean", {}),
            (2, {}, "mut_inf", {}, "mutInf", "mean", {}),
            (2, {}, "attr_conc", {}, "attrConc", "sd", {}),
            (2, {}, "attr_ent", {}, "attrEnt", "sd", {}),
            (2, {}, "class_conc", {}, "classConc", "sd", {}),
            (2, {}, "joint_ent", {}, "jointEnt", "sd", {}),
            (2, {}, "mut_inf", {}, "mutInf", "sd", {}),
            (2, {}, "attr_conc", {}, "attrConc", "histogram", {}),
            (2, {}, "attr_ent", {}, "attrEnt", "histogram", {}),
            (2, {}, "class_conc", {}, "classConc", "histogram", {}),
            (2, {}, "joint_ent", {}, "jointEnt", "histogram", {}),
            (2, {}, "mut_inf", {}, "mutInf", "histogram", {}),
            (2, {}, "attr_ent", {}, "attrEnt", "skewness", {}),
            (2, {}, "joint_ent", {}, "jointEnt", "min", {}),
            (2, {}, "mut_inf", {}, "mutInf", "max", {}),
            (1, {
                "transform_num": True,
            }, "eq_num_attr", {}, "eqNumAttr", "mean", {
                "transform": True,
            }),
            (0, {
                "transform_num": True,
            }, "ns_ratio", {}, "nsRatio", "all", {
                "transform": True,
            }),
            (1, {
                "transform_num": True,
            }, "attr_conc", {}, "attrConc", "mean", {
                "transform": True,
            }),
            (0, {
                "transform_num": True,
            }, "joint_ent", {}, "jointEnt", "mean", {
                "transform": True,
            }),
            (0, {
                "transform_num": True,
            }, "mut_inf", {}, "mutInf", "mean", {
                "transform": True,
            }),
            (1, {
                "transform_num": True,
            }, "mut_inf", {}, "mutInf", "mean", {
                "transform": True,
            }),
            (0, {
                "transform_num": True,
            }, "attr_ent", {}, "attrEnt", "sd", {
                "transform": True,
            }),
            (1, {
                "transform_num": True,
            }, "attr_ent", {}, "attrEnt", "sd", {
                "transform": True,
            }),
            (1, {
                "transform_num": True,
            }, "attr_ent", {}, "attrEnt", None, {
                "transform": True,
            }),
        ),
    )
    def test_ft_methods_info_theory(self, dt_id, fit_args_py, feat_name_py,
                                    method_args_py, feat_name_r, summary_name,
                                    fit_args_r):
        """Compare metafeat. values against correspondent from R mfe."""
        dataset = context.DATASET_LIST[dt_id]

        ind_targ, ind_attr = context.get_col_indexes(dataset)

        res_mfe_py = context.get_val_py(dataset, feat_name_py, ind_attr,
                                        ind_targ, method_args_py, fit_args_py,
                                        summary_name)

        res_mfe_r = context.get_val_r(dataset, feat_name_r, ind_attr, ind_targ,
                                      summary_name, "infotheo", **fit_args_r)

        assert context.compare_results(res_mfe_py, res_mfe_r, diff_factor=0.05)
