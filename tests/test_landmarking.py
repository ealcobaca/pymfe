"""Test module for Landmarking class metafeatures."""
import pytest

from pymfe.mfe import MFE
from tests.utils import load_xy
import numpy as np

GNAME = "landmarking"


class TestLandmarking():
    """TestClass dedicated to test Landmarking metafeatures."""

    @pytest.mark.parametrize(
        "dt_id, ft_name, exp_value, precompute, sample_size",
        [
            ###################
            # Mixed data
            ###################
            (0, 'best_node', [0.6, 0.116534315], False, 1.0),
            (0, 'elite_nn', [0.55833334, 0.15239853], False, 1.0),
            (0, 'linear_discr', [0.6666667, 0.20786986], False, 1.0),
            (0, 'naive_bayes', [0.6166667, 0.15811387], False, 1.0),
            (0, 'one_nn', [0.525, 0.07905694], False, 1.0),
            (0, 'worst_node', [0.5, 0.0], False, 1.0),
            (0, 'random_node', [0.5, 0.0], False, 1.0),
            (0, 'best_node', [0.6, 0.116534315], True, 1.0),
            (0, 'elite_nn', [0.55833334, 0.15239853], True, 1.0),
            (0, 'linear_discr', [0.6666667, 0.20786986], True, 1.0),
            (0, 'naive_bayes', [0.6166667, 0.15811387], True, 1.0),
            (0, 'one_nn', [0.525, 0.07905694], True, 1.0),
            (0, 'worst_node', [0.5, 0.0], True, 1.0),
            (0, 'random_node', [0.5, 0.0], True, 1.0),
            ###################
            # Categorical data
            ###################
            (1, 'best_node', [0.4899618, 0.12725325], False, 1.0),
            (1, 'elite_nn', [0.56567013, 0.080344684], False, 1.0),
            (1, 'linear_discr', [0.890417, 0.07505906], False, 1.0),
            (1, 'naive_bayes', [0.60906017, 0.09429], False, 1.0),
            (1, 'one_nn', [0.78222924, 0.056062974], False, 1.0),
            (1, 'worst_node', [0.4896993, 0.1311027], False, 1.0),
            (1, 'random_node', [0.56796771, 0.10101075], False, 1.0),
            (1, 'best_node', [0.4899618, 0.12725325], True, 1.0),
            (1, 'elite_nn', [0.56567013, 0.080344684], True, 1.0),
            (1, 'linear_discr', [0.890417, 0.07505906], True, 1.0),
            (1, 'naive_bayes', [0.60906017, 0.09429], True, 1.0),
            (1, 'one_nn', [0.78222924, 0.056062974], True, 1.0),
            (1, 'worst_node', [0.4896993, 0.1311027], True, 1.0),
            (1, 'random_node', [0.56796771, 0.10101075], True, 1.0),
            ###################
            # Numerical data
            ###################
            (2, 'best_node', [0.6666666, 6.282881e-08], False, 1.0),
            (2, 'elite_nn', [0.88, 0.061262432], False, 1.0),
            (2, 'linear_discr', [0.98, 0.044996567], False, 1.0),
            (2, 'naive_bayes', [0.9533334, 0.044996567], False, 1.0),
            (2, 'one_nn', [0.96000004, 0.056218266], False, 1.0),
            (2, 'worst_node', [0.6, 0.08888889], False, 1.0),
            (2, 'random_node', [0.66666663, 6.2828811e-08], False, 1.0),
            (2, 'best_node', [0.6666666, 6.282881e-08], True, 1.0),
            (2, 'elite_nn', [0.88, 0.061262432], True, 1.0),
            (2, 'linear_discr', [0.98, 0.044996567], True, 1.0),
            (2, 'naive_bayes', [0.9533334, 0.044996567], True, 1.0),
            (2, 'one_nn', [0.96000004, 0.056218266], True, 1.0),
            (2, 'worst_node', [0.6, 0.08888889], True, 1.0),
            (2, 'random_node', [0.66666663, 6.2828811e-08], True, 1.0),
            #######################################
            # Numerical data - Relative Landmarking
            #######################################
            (2, 'best_node', [0.6666666, 6.282881e-08], True, 0.5),
            (2, 'elite_nn', [0.93333328, 0.089962654], True, 0.5),
            (2, 'linear_discr', [0.9722222, 0.06000686], True, 0.5),
            (2, 'naive_bayes', [0.94444448, 0.10798059], True, 0.5),
            (2, 'one_nn', [1.0, 0.0], True, 0.5),
            (2, 'worst_node', [0.65555555, 0.035136417], True, 0.5),
            (2, 'random_node', [0.6666666, 6.282881e-08], True, 0.5),
            (2, 'best_node', [0.6666666, 6.282881e-08], False, 0.5),
            (2, 'elite_nn', [0.93333328, 0.089962654], False, 0.5),
            (2, 'linear_discr', [0.9722222, 0.06000686], False, 0.5),
            (2, 'naive_bayes', [0.94444448, 0.10798059], False, 0.5),
            (2, 'one_nn', [1.0, 0.0], False, 0.5),
            (2, 'worst_node', [0.65555555, 0.035136417], False, 0.5),
            (2, 'random_node', [0.6666666, 6.282881e-08], False, 0.5),
        ])
    def test_ft_methods_landmarking(self, dt_id, ft_name, exp_value,
                                    precompute, sample_size):
        """Function to test each meta-feature belongs to landmarking group.
        """
        precomp_group = "landmarking" if precompute else None

        X, y = load_xy(dt_id)
        mfe = MFE(
            groups=["landmarking"],
            features=[ft_name],
            sample_size=sample_size,
            random_state=1234)

        mfe.fit(X.values, y.values, precomp_groups=precomp_group)

        value = mfe.extract()[1]

        if exp_value is np.nan:
            assert value[0] is exp_value

        else:
            assert np.allclose(value, exp_value)

    @pytest.mark.parametrize(
        "dt_id, summary, precompute, sample_size, exp_value",
        [
            #######################################
            # Mean Relative Landmarking
            #######################################
            (0, "mean", False, 1.0, [5.0, 4.0, 7.0, 6.0, 3.0, 1.5, 1.5]),
            (0, "mean", True,  1.0, [5.0, 4.0, 7.0, 6.0, 3.0, 1.5, 1.5]),
            (1, "mean", False, 1.0, [2.0, 3.0, 7.0, 5.0, 6.0, 4.0, 1.0]),
            (1, "mean", True,  1.0, [2.0, 3.0, 7.0, 5.0, 6.0, 4.0, 1.0]),
            (2, "mean", False, 1.0, [2.5, 4.0, 7.0, 5.0, 6.0, 2.5, 1.0]),
            (2, "mean", True,  1.0, [2.5, 4.0, 7.0, 5.0, 6.0, 2.5, 1.0]),

            #######################################
            # Mean Relative Subsampling Landmarking
            #######################################
            (0, "mean", False, 0.5, [5.0, 7.0, 2.0, 6.0, 1.0, 4.0, 3.0]),
            (0, "mean", True,  0.5, [5.0, 7.0, 2.0, 6.0, 1.0, 4.0, 3.0]),
            (1, "mean", False, 0.5, [5.0, 3.0, 7.0, 4.0, 6.0, 2.0, 1.0]),
            (1, "mean", True,  0.5, [5.0, 3.0, 7.0, 4.0, 6.0, 2.0, 1.0]),
            (2, "mean", False, 0.5, [2.5, 4.0, 6.0, 5.0, 7.0, 2.5, 1.0]),
            (2, "mean", True,  0.5, [2.5, 4.0, 6.0, 5.0, 7.0, 2.5, 1.0]),

            #######################################
            # Std Relative Landmarking
            #######################################
            (0, "sd", False, 1.0, [4.0, 5.0, 7.0, 6.0, 3.0, 1.5, 1.5]),
            (0, "sd", True,  1.0, [4.0, 5.0, 7.0, 6.0, 3.0, 1.5, 1.5]),
            (1, "sd", False, 1.0, [6.0, 3.0, 2.0, 4.0, 1.0, 5.0, 7.0]),
            (1, "sd", True,  1.0, [6.0, 3.0, 2.0, 4.0, 1.0, 5.0, 7.0]),
            (2, "sd", False, 1.0, [1.5, 6.0, 3.5, 3.5, 5.0, 1.5, 7.0]),
            (2, "sd", True,  1.0, [1.5, 6.0, 3.5, 3.5, 5.0, 1.5, 7.0]),

            #######################################
            # Std Relative Subsampling Landmarking
            #######################################
            (0, "sd", False, 0.5, [6.0, 5.0, 4.0, 7.0, 3.0, 1.0, 2.0]),
            (0, "sd", True,  0.5, [6.0, 5.0, 4.0, 7.0, 3.0, 1.0, 2.0]),
            (1, "sd", False, 0.5, [3.0, 7.0, 1.0, 4.0, 2.0, 6.0, 5.0]),
            (1, "sd", True,  0.5, [3.0, 7.0, 1.0, 4.0, 2.0, 6.0, 5.0]),
            (2, "sd", False, 0.5, [2.5, 6.0, 5.0, 7.0, 1.0, 2.5, 4.0]),
            (2, "sd", True,  0.5, [2.5, 6.0, 5.0, 7.0, 1.0, 2.5, 4.0]),
        ])
    def test_ft_method_relative(self, dt_id, summary, precompute,
                                sample_size, exp_value):
        """Test relative and subsampling relative landmarking."""
        precomp_group = "relative" if precompute else None

        X, y = load_xy(dt_id)
        mfe = MFE(
            groups=["relative"],
            summary=summary,
            sample_size=sample_size,
            random_state=1234)

        mfe.fit(X.values, y.values, precomp_groups=precomp_group)

        _, vals = mfe.extract()

        assert np.allclose(vals, exp_value)

    @pytest.mark.parametrize(
        "summary, dt_id",
        [
            ("mean", 0),
            ("sd", 0),
            ("histogram", 0),
            ("quantiles", 0),
            ("max", 0),
            ("mean", 1),
            ("sd", 1),
            ("histogram", 1),
            ("quantiles", 1),
            ("max", 1),
            ("mean", 2),
            ("sd", 2),
            ("histogram", 2),
            ("quantiles", 2),
            ("max", 2),
        ])
    def test_relative_correctness(self, summary, dt_id):
        """Test if the metafeatures postprocessed by rel. land. are correct."""
        X, y = load_xy(dt_id)
        mfe = MFE(
            groups="all",
            summary=summary,
            sample_size=0.5,
            random_state=1234)

        mfe.fit(X.values, y.values)

        names, _ = mfe.extract()

        target_mtf = mfe.valid_metafeatures(groups="landmarking")

        relative_names = {
            name.split(".")[0]
            for name in names
            if name.rfind(".relative") != -1
        }

        assert not set(relative_names).symmetric_difference(target_mtf)
