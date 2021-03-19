"""Test module for concept metafeatures."""
import pytest

from pymfe.mfe import MFE
from tests.utils import load_xy
import numpy as np

GNAME = "concept"


class TestConcept:
    """TestClass dedicated to test concept metafeatures."""

    @pytest.mark.parametrize(
        "dt_id, ft_name, exp_value, precompute",
        [
            ###################
            # Mixed data
            ###################
            (0, "cohesiveness", [10.055, 1.1869723], True),
            (0, "conceptvar", [0.5389795, 0.010408287], True),
            (0, "impconceptvar", [5.275, 0.59225446], True),
            (0, "wg_dist", [1.4762982, 0.07838156], True),
            (0, "cohesiveness", [10.055, 1.1869723], False),
            (0, "conceptvar", [0.5389795, 0.010408287], False),
            (0, "impconceptvar", [5.275, 0.59225446], False),
            (0, "wg_dist", [1.4762982, 0.07838156], False),
            ###################
            # Categorical data
            ###################
            (1, "cohesiveness", [306.5352, 48.729893], True),
            (1, "conceptvar", [0.47566572, 0.036749393], True),
            (1, "impconceptvar", [146.3541, 25.366209], True),
            (1, "wg_dist", [2.9002495, 0.21794802], True),
            (1, "cohesiveness", [306.5352, 48.729893], False),
            (1, "conceptvar", [0.47566572, 0.036749393], False),
            (1, "impconceptvar", [146.3541, 25.366209], False),
            (1, "wg_dist", [2.9002495, 0.21794802], False),
            ###################
            # Numerical data
            ###################
            (2, "cohesiveness", [67.12, 5.3592987], True),
            (2, "conceptvar", [0.4956224, 0.07772438], True),
            (2, "impconceptvar", [42.626667, 5.358048], True),
            (2, "wg_dist", [0.46218988, 0.05621875], True),
            (2, "cohesiveness", [67.12, 5.3592987], False),
            (2, "conceptvar", [0.4956224, 0.07772438], False),
            (2, "impconceptvar", [42.626667, 5.358048], False),
            (2, "wg_dist", [0.46218988, 0.05621875], False),
        ],
    )
    def test_ft_methods_complexity(
        self, dt_id, ft_name, exp_value, precompute
    ):
        """Function to test each meta-feature belongs to concept group."""
        precomp_group = GNAME if precompute else None

        X, y = load_xy(dt_id)
        mfe = MFE(groups=[GNAME], features=[ft_name], random_state=1234)

        mfe.fit(X.values, y.values, precomp_groups=precomp_group)

        value = mfe.extract()[1]

        if exp_value is np.nan:
            assert value[0] is exp_value
        else:
            assert np.allclose(value, exp_value, equal_nan=True)

    @pytest.mark.parametrize(
        "dt_id, exp_value, precompute",
        [
            ###################
            # Mixed data
            ###################
            (0, [10.055, 0.5389795, 5.275, 1.4762982], False),
            (0, [10.055, 0.5389795, 5.275, 1.4762982], True),
            ###################
            # Numerical data
            ###################
            (2, [67.12, 0.4956224, 42.626667, 0.46218988], False),
            (2, [67.12, 0.4956224, 42.626667, 0.46218988], True),
        ],
    )
    def test_integration_concept(self, dt_id, exp_value, precompute):
        """Function to test each all concept meta-features."""
        precomp_group = GNAME if precompute else None

        X, y = load_xy(dt_id)
        mfe = MFE(groups=[GNAME], summary="mean", random_state=1234)

        mfe.fit(X.values, y.values, precomp_groups=precomp_group)

        value = mfe.extract()[1]

        assert np.allclose(value, exp_value, equal_nan=True)
