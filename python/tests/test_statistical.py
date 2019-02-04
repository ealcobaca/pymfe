"""Test module for Statistical class metafeatures."""
import pytest

import context


class TestInfoTheory:
    """TestClass dedicated to test Statistical metafeatures."""

    @pytest.mark.parametrize(
        ("dt_id, fit_args_py, feat_name_py,"
         "method_args_py, feat_name_r, summary_name, fit_args_r"),
        (
            (0, {}, "gravity", {}, "gravity", "all", {}),
            (0, {}, "nr_cor_attr", {}, "nrCorAttr", "all", {}),
            (0, {}, "nr_norm", {}, "nrNorm", "all", {}),
            (0, {}, "nr_outliers", {}, "nrOutliers", "all", {}),
            (0, {}, "sd_ratio", {}, "sdRatio", "all", {}),
            (0, {}, "w_lambda", {}, "wLambda", "all", {}),
            (1, {}, "w_lambda", {}, "wLambda", "all", {
                "transform": False,
            }),
            (1, {"transform_cat": True}, "w_lambda", {},
                "wLambda", "all", {
                "transform": True,
            }),
            (0, {}, "cor", {}, "cor", "mean", {}),
            (0, {}, "cov", {}, "cov", "mean", {}),
            (0, {}, "eigenvalues", {}, "eigenvalues", "mean", {}),
            (0, {}, "g_mean", {}, "gMean", "mean", {}),
            (0, {}, "h_mean", {}, "hMean", "mean", {}),
            (0, {}, "iq_range", {}, "iqRange", "mean", {}),
            (0, {}, "kurtosis", {}, "kurtosis", "mean", {}),
            (0, {}, "mad", {}, "mad", "mean", {}),
            (0, {}, "max", {}, "max", "mean", {}),
            (0, {}, "mean", {}, "mean", "mean", {}),
            (0, {}, "median", {}, "median", "mean", {}),
            (0, {}, "min", {}, "min", "mean", {}),
            (0, {}, "range", {}, "range", "mean", {}),
            (0, {}, "sd", {}, "sd", "mean", {}),
            (0, {}, "skewness", {}, "skewness", "mean", {}),
            (0, {}, "sparsity", {}, "sparsity", "mean", {}),
            (0, {}, "t_mean", {}, "tMean", "mean", {}),
            (0, {}, "var", {}, "var", "mean", {}),
            (0, {}, "cor", {}, "cor", "histogram", {}),
            (0, {}, "cov", {}, "cov", "max", {}),
            (0, {}, "eigenvalues", {}, "eigenvalues", "min", {}),
            (0, {}, "g_mean", {}, "gMean", "sd", {}),
            (0, {}, "h_mean", {}, "hMean", "var", {}),
            (0, {}, "kurtosis", {}, "kurtosis", "median", {}),
            (0, {}, "mad", {}, "mad", "kurtosis", {}),
            (0, {}, "max", {}, "max", "skewness", {}),
            (0, {}, "median", {}, "median", "sd", {}),
            (0, {}, "min", {}, "min", "kurtosis", {}),
            (0, {}, "range", {}, "range", "median", {}),
            (0, {}, "sd", {}, "sd", "max", {}),
            (0, {}, "skewness", {}, "skewness", "min", {}),
            (0, {}, "sparsity", {}, "sparsity", "var", {}),
            (0, {}, "t_mean", {}, "tMean", "histogram", {}),
            (0, {}, "var", {}, "var", "sd", {}),
            (0, {}, "can_cor", {}, "canCor", "mean", {}),
            (0, {}, "nr_disc", {}, "nrDisc", "all", {}),
            (2, {
                "transform_cat": True,
            }, "skewness", {}, "skewness", "mean", {
                "transform": True,
            }),
            (2, {
                "transform_cat": True,
            }, "kurtosis", {}, "kurtosis", "mean", {
                "transform": True,
            }),
            (1, {
                "transform_cat": True,
            }, "kurtosis", {}, "kurtosis", "mean", {
                "transform": True,
            }),
            (1, {
                "transform_cat": True,
            }, "mad", {}, "mad", "mean", {
                "transform": True,
            }),
            (2, {
                "transform_cat": True,
            }, "h_mean", {}, "hMean", "sd", {
                "transform": True,
            }),
            (1, {
                "transform_cat": True,
            }, "eigenvalues", {}, "eigenvalues", "mean", {
                "transform": True,
            }),
            (2, {
                "transform_cat": True,
            }, "eigenvalues", {}, "eigenvalues", "mean", {
                "transform": True,
            }),
            (2, {
                "transform_cat": True,
            }, "gravity", {}, "gravity", "all", {
                "transform": True,
            }),
            (2, {
                "transform_cat": True,
            }, "nr_cor_attr", {}, "nrCorAttr", "all", {
                "transform": True,
            }),
            (2, {
                "transform_cat": True,
            }, "nr_norm", {}, "nrNorm", "all", {
                "transform": True,
            }),
            (2, {
                "transform_cat": True,
            }, "nr_outliers", {}, "nrOutliers", "all", {
                "transform": True,
            }),
            (2, {
                "transform_cat": True,
            }, "sd_ratio", {}, "sdRatio", "all", {
                "transform": True,
            }),
            (2, {
                "transform_cat": True,
            }, "w_lambda", {}, "wLambda", "all", {
                "transform": True,
            }),
        ),
    )
    def test_ft_methods_statistical(self, dt_id, fit_args_py, feat_name_py,
                                    method_args_py, feat_name_r, summary_name,
                                    fit_args_r):
        """Compare metafeat. values against correspondent from R mfe."""
        dataset = context.DATASET_LIST[dt_id]

        ind_targ, ind_attr = context.get_col_indexes(dataset)

        res_mfe_py = context.get_val_py(dataset, feat_name_py, ind_attr,
                                        ind_targ, method_args_py, fit_args_py,
                                        summary_name)

        res_mfe_r = context.get_val_r(dataset, feat_name_r, ind_attr, ind_targ,
                                      summary_name, "statistical",
                                      **fit_args_r)

        assert context.compare_results(res_mfe_py, res_mfe_r, diff_factor=0.05)
