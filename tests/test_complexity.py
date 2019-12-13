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
            (0, 'c1', [1.0], True),
            (0, 'c2', [0.0], True),
            (0, 'f3', [0.76, np.nan], True),
            (0, 'f4', [0.66, np.nan], True),
            (0, 'l2', [0.24, np.nan], True),
            (0, 'n1', [1.0], True),
            (0, 'n4', [0.48], True),
            (0, 't2', [0.08], True),
            (0, 't3', [0.02], True),
            (0, 't4', [0.09090909], True),
            (0, 'c1', [1.0], False),
            (0, 'c2', [0.0], False),
            (0, 'f3', [0.76, np.nan], False),
            (0, 'f4', [0.66, np.nan], False),
            (0, 'l2', [0.24, np.nan], False),
            (0, 'n1', [1.0], False),
            (0, 'n4', [0.48], False),
            (0, 't2', [0.08], False),
            (0, 't3', [0.02], False),
            (0, 't4', [0.09090909], False),
            ###################
            # Categorical data
            ###################
            (1, 'c1', [0.998575538], True),
            (1, 'c2', [0.003940366], True),
            (1, 'f3', [0.8172716, np.nan], True),
            (1, 'f4', [0.5669587, np.nan], True),
            (1, 'l2', [0.025969962, np.nan], True),
            (1, 'n1', [0.25563204], True),
            (1, 'n4', [0.13673341], True),
            (1, 't2', [0.011264080100125156], True),
            (1, 't3', [0.00750938], True),
            (1, 't4', [0.63157894], True),
            (1, 'c1', [0.998575538], False),
            (1, 'c2', [0.003940366], False),
            (1, 'f3', [0.8172716, np.nan], False),
            (1, 'f4', [0.5669587, np.nan], False),
            (1, 'l2', [0.025969962, np.nan], False),
            (1, 'n1', [0.25563204], False),
            (1, 'n4', [0.13673341], False),
            (1, 't2', [0.011264080100125156], False),
            (1, 't3', [0.00750938], False),
            (1, 't4', [0.63157894], False),
            ###################
            # Numerical data
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
            (2, 'c2', [0.0], False),
            (2, 'f3', [0.123333, 0.213620], False),
            (2, 'f4', [0.043333, 0.075055], False),
            (2, 'l2', [0.013333333, 0.023094011], False),
            (2, 'n1', [0.1], False),
            (2, 'n4', [0.01333333], False),
            (2, 't2', [0.02666667], False),
            (2, 't3', [0.01333333], False),
            (2, 't4', [0.5], False),
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

        if exp_value is np.nan:
            assert value[0] is exp_value
        else:
            assert np.allclose(value, exp_value, equal_nan=True)
