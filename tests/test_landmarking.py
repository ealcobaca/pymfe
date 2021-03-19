"""Test module for landmarking class metafeatures."""
import pytest

from pymfe.mfe import MFE
from tests.utils import load_xy
import numpy as np

GNAME = "landmarking"


class TestLandmarking:
    """TestClass dedicated to test landmarking metafeatures."""

    @pytest.mark.parametrize(
        "dt_id, ft_name, exp_value, precompute, lm_sample_frac",
        [
            ###################
            # Mixed data
            ###################
            (0, "best_node", [0.64, 0.15776213], False, 1.0),
            (0, "elite_nn", [0.56000006, 0.15776213], False, 1.0),
            (0, "linear_discr", [0.52, 0.21499355], False, 1.0),
            (0, "naive_bayes", [0.66, 0.13498971], False, 1.0),
            (0, "one_nn", [0.26000002, 0.13498971], False, 1.0),
            (0, "random_node", [0.4, 0.0], False, 1.0),
            (0, "worst_node", [0.4, 0.0], False, 1.0),
            (0, "best_node", [0.64, 0.15776213], True, 1.0),
            (0, "elite_nn", [0.56000006, 0.15776213], True, 1.0),
            (0, "linear_discr", [0.52, 0.21499355], True, 1.0),
            (0, "naive_bayes", [0.66, 0.13498971], True, 1.0),
            (0, "one_nn", [0.26000002, 0.13498971], True, 1.0),
            (0, "worst_node", [0.4, 0.0], True, 1.0),
            (0, "random_node", [0.4, 0.0], True, 1.0),
            ###################
            # Categorical data
            ###################
            (1, "best_node", [0.4896346, 0.12722623], False, 1.0),
            (1, "elite_nn", [0.5656701, 0.080348104], False, 1.0),
            (1, "linear_discr", [0.89012927, 0.07484206], False, 1.0),
            (1, "naive_bayes", [0.6117996, 0.098759025], False, 1.0),
            (1, "one_nn", [0.7822502, 0.055724006], False, 1.0),
            (1, "random_node", [0.56793106, 0.10140118], False, 1.0),
            (1, "worst_node", [0.48092183, 0.13144408], False, 1.0),
            (1, "best_node", [0.4896346, 0.12722623], True, 1.0),
            (1, "elite_nn", [0.5656701, 0.080348104], True, 1.0),
            (1, "linear_discr", [0.89012927, 0.07484206], True, 1.0),
            (1, "naive_bayes", [0.6117996, 0.098759025], True, 1.0),
            (1, "one_nn", [0.7822502, 0.055724006], True, 1.0),
            (1, "worst_node", [0.48092183, 0.13144408], True, 1.0),
            (1, "random_node", [0.56793106, 0.10140118], True, 1.0),
            ###################
            # Numerical data
            ###################
            (2, "best_node", [0.6666666, 6.282881e-08], False, 1.0),
            (2, "elite_nn", [0.88, 0.061262432], False, 1.0),
            (2, "linear_discr", [0.98, 0.044996567], False, 1.0),
            (2, "naive_bayes", [0.9533334, 0.044996567], False, 1.0),
            (2, "one_nn", [0.96000004, 0.056218266], False, 1.0),
            (2, "random_node", [0.66666663, 6.2828811e-08], False, 1.0),
            (2, "worst_node", [0.6, 0.08888889], False, 1.0),
            (2, "best_node", [0.6666666, 6.282881e-08], True, 1.0),
            (2, "elite_nn", [0.88, 0.061262432], True, 1.0),
            (2, "linear_discr", [0.98, 0.044996567], True, 1.0),
            (2, "naive_bayes", [0.9533334, 0.044996567], True, 1.0),
            (2, "one_nn", [0.96000004, 0.056218266], True, 1.0),
            (2, "worst_node", [0.6, 0.08888889], True, 1.0),
            (2, "random_node", [0.66666663, 6.2828811e-08], True, 1.0),
            #######################################
            # Numerical data - Relative landmarking
            #######################################
            (2, "best_node", [0.5982143, 0.02823461], True, 0.5),
            (2, "elite_nn", [0.9196428, 0.14803368], True, 0.5),
            (2, "linear_discr", [0.9732143, 0.056625884], True, 0.5),
            (2, "naive_bayes", [0.9464285, 0.09105392], True, 0.5),
            (2, "one_nn", [1.0, 0.0], True, 0.5),
            (2, "random_node", [0.5982143, 0.02823461], True, 0.5),
            (2, "worst_node", [0.5696429, 0.1032528], True, 0.5),
            (2, "best_node", [0.5982143, 0.02823461], False, 0.5),
            (2, "elite_nn", [0.9196428, 0.14803368], False, 0.5),
            (2, "linear_discr", [0.9732143, 0.056625884], False, 0.5),
            (2, "naive_bayes", [0.9464285, 0.09105392], False, 0.5),
            (2, "one_nn", [1.0, 0.0], False, 0.5),
            (2, "worst_node", [0.5696429, 0.1032528], False, 0.5),
            (2, "random_node", [0.5982143, 0.02823461], False, 0.5),
        ],
    )
    def test_ft_methods_landmarking(
        self, dt_id, ft_name, exp_value, precompute, lm_sample_frac
    ):
        """Function to test each meta-feature belongs to landmarking group."""
        precomp_group = GNAME if precompute else None

        X, y = load_xy(dt_id)
        mfe = MFE(
            groups=[GNAME],
            features=[ft_name],
            lm_sample_frac=lm_sample_frac,
            random_state=1234,
        )

        mfe.fit(X.values, y.values, precomp_groups=precomp_group)

        value = mfe.extract()[1]

        if exp_value is np.nan:
            assert value[0] is exp_value

        else:
            assert np.allclose(value, exp_value, equal_nan=True, atol=1.0e-6)

    @pytest.mark.parametrize(
        "dt_id, exp_value, precompute, lm_sample_frac",
        [
            ###################
            # Mixed data
            ###################
            (
                0,
                [0.64, 0.56000006, 0.52, 0.66, 0.26000002, 0.4, 0.4],
                False,
                1.0,
            ),
            (
                0,
                [0.64, 0.56000006, 0.52, 0.66, 0.26000002, 0.4, 0.4],
                True,
                1.0,
            ),
            ###################
            # Numerical data
            ###################
            (
                2,
                [
                    0.6666666,
                    0.88,
                    0.98,
                    0.9533334,
                    0.96000004,
                    0.66666663,
                    0.6,
                ],
                False,
                1.0,
            ),
            (
                2,
                [
                    0.6666666,
                    0.88,
                    0.98,
                    0.9533334,
                    0.96000004,
                    0.66666663,
                    0.6,
                ],
                True,
                1.0,
            ),
            #######################################
            # Numerical data - Relative landmarking
            #######################################
            (
                2,
                [
                    0.5982143,
                    0.9196428,
                    0.9732143,
                    0.9464285,
                    1.0,
                    0.5982143,
                    0.5696429,
                ],
                False,
                0.5,
            ),
            (
                2,
                [
                    0.5982143,
                    0.9196428,
                    0.9732143,
                    0.9464285,
                    1.0,
                    0.5982143,
                    0.5696429,
                ],
                True,
                0.5,
            ),
        ],
    )
    def test_integration_landmarking(
        self, dt_id, exp_value, precompute, lm_sample_frac
    ):
        """Function to test all landmarking meta-features."""
        precomp_group = GNAME if precompute else None

        X, y = load_xy(dt_id)
        mfe = MFE(
            groups=[GNAME],
            summary="mean",
            lm_sample_frac=lm_sample_frac,
            random_state=1234,
        )

        mfe.fit(X.values, y.values, precomp_groups=precomp_group)

        value = mfe.extract()[1]

        assert np.allclose(value, exp_value, equal_nan=True, atol=1.0e-6)
