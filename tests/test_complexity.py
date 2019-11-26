"""Test module for Complexity class metafeatures."""
import pytest

from pymfe.mfe import MFE
from tests.utils import load_xy
import numpy as np

GNAME = "complexity"


class TestLandmarking():
    """TestClass dedicated to test Complexity metafeatures."""

    @pytest.mark.parametrize(
        "dt_id, ft_name, exp_value, precompute",
        [
            ###################
            # Mixed data
            ###################
            (2, 'c1', [0.999999], True),
            (2, 'c2', [0.0], True),
            (2, 'f3', [0.123333, 0.213620], True),
            (2, 'f4', [0.043333, 0.075055], True),
            (2, 'l2', [0.013333333, 0.023094011], True),
            (2, 'n1', [0.1], True),
            (2, 'n4', [0.01333333], True),
            (2, 't2', [0.02666667], True),
            (2, 't3', [0.01333333], True),
            (2, 't4', [0.5], True),
            (2, 'c1', [0.999999], False),
            # (2, 'c2', [0.0], False),
            # (2, 'f3', [0.123333, 0.213620], False),
            # (2, 'f4', [0.043333, 0.075055], False),
            # (2, 'l2', [0.013333333, 0.023094011], False),
            # (2, 'n1', [0.1], False),
            # (2, 'n4', [0.01333333], False),
            # (2, 't2', [0.02666667], False),
            # (2, 't3', [0.01333333], False),
            # (2, 't4', [0.5], False)
            ###################
            # Categorical data
            ###################
            # (1, 'best_node', [0.4899618, 0.12725325], False, 1.0),
            # (1, 'elite_nn', [0.56567013, 0.080344684], False, 1.0),
            # (1, 'linear_discr', [0.890417, 0.07505906], False, 1.0),
            # (1, 'naive_bayes', [0.60906017, 0.09429], False, 1.0),
            # (1, 'one_nn', [0.78222924, 0.056062974], False, 1.0),
            # (1, 'worst_node', [0.4896993, 0.1311027], False, 1.0),
            # (1, 'random_node', [0.56796771, 0.10101075], False, 1.0),
            # (1, 'best_node', [0.4899618, 0.12725325], True, 1.0),
            # (1, 'elite_nn', [0.56567013, 0.080344684], True, 1.0),
            # (1, 'linear_discr', [0.890417, 0.07505906], True, 1.0),
            # (1, 'naive_bayes', [0.60906017, 0.09429], True, 1.0),
            # (1, 'one_nn', [0.78222924, 0.056062974], True, 1.0),
            # (1, 'worst_node', [0.4896993, 0.1311027], True, 1.0),
            # (1, 'random_node', [0.56796771, 0.10101075], True, 1.0),
            ###################
            # Numerical data
            ###################
            # (2, 'best_node', [0.6666666, 6.282881e-08], False, 1.0),
            # (2, 'elite_nn', [0.88, 0.061262432], False, 1.0),
            # (2, 'linear_discr', [0.98, 0.044996567], False, 1.0),
            # (2, 'naive_bayes', [0.9533334, 0.044996567], False, 1.0),
            # (2, 'one_nn', [0.96000004, 0.056218266], False, 1.0),
            # (2, 'worst_node', [0.6, 0.08888889], False, 1.0),
            # (2, 'random_node', [0.66666663, 6.2828811e-08], False, 1.0),
            # (2, 'best_node', [0.6666666, 6.282881e-08], True, 1.0),
            # (2, 'elite_nn', [0.88, 0.061262432], True, 1.0),
            # (2, 'linear_discr', [0.98, 0.044996567], True, 1.0),
            # (2, 'naive_bayes', [0.9533334, 0.044996567], True, 1.0),
            # (2, 'one_nn', [0.96000004, 0.056218266], True, 1.0),
            # (2, 'worst_node', [0.6, 0.08888889], True, 1.0),
            # (2, 'random_node', [0.66666663, 6.2828811e-08], True, 1.0),
        ])
    def test_ft_methods_complexity(self, dt_id, ft_name, exp_value,
                                   precompute):
        """Function to test each meta-feature belongs to complexity group.
        """
        precomp_group = GNAME if precompute else None

        X, y = load_xy(dt_id)
        mfe = MFE(
            groups=["complexity"],
            features=[ft_name],
            random_state=1234)

        mfe.fit(X.values, y.values, precomp_groups=precomp_group)

        value = mfe.extract()[1]
        print(value)

        if exp_value is np.nan:
            assert value[0] is exp_value

        else:
            assert np.allclose(value, exp_value)
