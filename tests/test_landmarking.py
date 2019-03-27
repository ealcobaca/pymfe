"""Test module for Landmarking class metafeatures."""
import pytest

from pymfe.mfe import MFE
from tests.utils import load_xy
import numpy as np

GNAME = "landmarking"


class TestLandmarking():
    """TestClass dedicated to test Landmarking metafeatures."""

    @pytest.mark.parametrize(
        "dt_id, ft_name, exp_value, precompute",
        [
            ###################
            # Mixed data
            ###################
            (0, 'best_node', [0.6, 0.116534315], False),
            (0, 'elite_nn', [0.55833334, 0.15239853], False),
            (0, 'linear_discr', [0.6666667, 0.20786986], False),
            (0, 'naive_bayes', [0.6166667, 0.15811387], False),
            (0, 'one_nn', [0.525, 0.07905694], False),
            (0, 'worst_node', [0.5, 0.0], False),
            (0, 'best_node', [0.6, 0.116534315], True),
            (0, 'elite_nn', [0.55833334, 0.15239853], True),
            (0, 'linear_discr', [0.6666667, 0.20786986], True),
            (0, 'naive_bayes', [0.6166667, 0.15811387], True),
            (0, 'one_nn', [0.525, 0.07905694], True),
            (0, 'worst_node', [0.5, 0.0], True),
            ###################
            # Categorical data
            ###################
            (1, 'best_node', [0.4899618, 0.12725325], False),
            (1, 'elite_nn', [0.56567013, 0.080344684], False),
            (1, 'linear_discr', [0.890417, 0.07505906], False),
            (1, 'naive_bayes', [0.60906017, 0.09429], False),
            (1, 'one_nn', [0.78222924, 0.056062974], False),
            (1, 'worst_node', [0.4896993, 0.1311027], False),
            (1, 'best_node', [0.4899618, 0.12725325], True),
            (1, 'elite_nn', [0.56567013, 0.080344684], True),
            (1, 'linear_discr', [0.890417, 0.07505906], True),
            (1, 'naive_bayes', [0.60906017, 0.09429], True),
            (1, 'one_nn', [0.78222924, 0.056062974], True),
            (1, 'worst_node', [0.4896993, 0.1311027], True),
            ###################
            # Numerical data
            ###################
            (2, 'best_node', [0.6666666, 6.282881e-08], False),
            (2, 'elite_nn', [0.88, 0.061262432], False),
            (2, 'linear_discr', [0.98, 0.044996567], False),
            (2, 'naive_bayes', [0.9533334, 0.044996567], False),
            (2, 'one_nn', [0.96000004, 0.056218266], False),
            (2, 'worst_node', [0.6, 0.08888889], False),
            (2, 'best_node', [0.6666666, 6.282881e-08], True),
            (2, 'elite_nn', [0.88, 0.061262432], True),
            (2, 'linear_discr', [0.98, 0.044996567], True),
            (2, 'naive_bayes', [0.9533334, 0.044996567], True),
            (2, 'one_nn', [0.96000004, 0.056218266], True),
            (2, 'worst_node', [0.6, 0.08888889], True),
        ])
    def test_ft_methods_landmarking(self, dt_id, ft_name, exp_value,
                                    precompute):
        """Function to test each meta-feature belongs to landmarking group.
        """
        precomp_group = "landmarking" if precompute else None

        X, y = load_xy(dt_id)
        mfe = MFE(
            groups=["landmarking"], features=[ft_name], random_state=1234)

        mfe.fit(X.values, y.values, precomp_groups=precomp_group)

        value = mfe.extract()[1]

        if exp_value is np.nan:
            assert value[0] is exp_value

        else:
            assert np.allclose(value, exp_value)
