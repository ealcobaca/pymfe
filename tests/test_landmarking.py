"""Test module for Landmarking class metafeatures."""
import pytest

from pymfe.mfe import MFE
from tests.utils import load_xy
import numpy as np

GNAME = "landmarking"


class TestLandmarking():
    """TestClass dedicated to test Landmarking metafeatures."""

    @pytest.mark.parametrize(
        "dt_id, ft_name, exp_value, precompute, size",
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
            (2, 'elite_nn', [0.9722222, 0.06000686], True, 0.5),
            (2, 'linear_discr', [0.9722222, 0.06000686], True, 0.5),
            (2, 'naive_bayes', [0.96111107, 0.06441677], True, 0.5),
            (2, 'one_nn', [0.95, 0.08466022], True, 0.5),
            (2, 'worst_node', [0.65, 0.052704636], True, 0.5),
            (2, 'random_node', [0.6666666, 6.282881e-08], True, 0.5),
        ])
    def test_ft_methods_landmarking(self, dt_id, ft_name, exp_value,
                                    precompute, size):
        """Function to test each meta-feature belongs to landmarking group.
        """
        precomp_group = "landmarking" if precompute else None

        X, y = load_xy(dt_id)
        mfe = MFE(
            groups=["landmarking"], features=[ft_name], size=size,
            random_state=1234)

        mfe.fit(X.values, y.values, precomp_groups=precomp_group)

        value = mfe.extract()[1]

        if exp_value is np.nan:
            assert value[0] is exp_value

        else:
            assert np.allclose(value, exp_value)
