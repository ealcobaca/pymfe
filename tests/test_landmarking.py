"""Test module for Landmarking class metafeatures."""
import pytest

from pymfe.mfe import MFE
from tests.utils import load_xy
import numpy as np

GNAME = "landmarking"


class TestLandmarking():
    """TestClass dedicated to test Landmarking metafeatures."""
    @pytest.mark.parametrize("dt_id, ft_name, exp_value", [
        ###################
        # Mixed data
        ###################
        (0, 'best_node', [0.6, 0.116534315]),
        (0, 'elite_nn', [0.55833334, 0.15239853]),
        (0, 'linear_discr', [0.6666667, 0.20786986]),
        (0, 'naive_bayes', [0.6166667, 0.15811387]),
        (0, 'one_nn', [0.525, 0.07905694]),
        (0, 'worst_node', [0.5, 0.0]),
        ###################
        # Categorical data
        ###################
        (1, 'best_node', [0.4899618, 0.12725325]),
        (1, 'elite_nn', [0.56567013, 0.080344684]),
        (1, 'linear_discr', [0.890417, 0.07505906]),
        (1, 'naive_bayes',[0.60906017, 0.09429]),
        (1, 'one_nn', [0.78222924, 0.056062974]),
        (1, 'worst_node', [0.4896993, 0.1311027]),
        ###################
        # Numerical data
        ###################
        (2, 'best_node', [0.6666666, 6.282881e-08]),
        (2, 'elite_nn', [0.88, 0.061262432]),
        (2, 'linear_discr', [0.98, 0.044996567]),
        (2, 'naive_bayes', [0.9533334, 0.044996567]),
        (2, 'one_nn', [0.96000004, 0.056218266]),
        (2, 'worst_node', [0.6, 0.08888889]),
    ])
    def test_ft_methods_landmarking(self, dt_id, ft_name, exp_value):
        """Function to test each meta-feature belongs to landmarking group.
        """
        X, y = load_xy(dt_id)
        mfe = MFE(groups=["landmarking"],
                  features=[ft_name], random_state=1234).fit(X.values, y.values)
        value = mfe.extract()[1]

        if exp_value is np.nan:
            assert value[0] is exp_value

        else:
            assert np.allclose(value, exp_value)
