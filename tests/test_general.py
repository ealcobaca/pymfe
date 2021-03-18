"""Test module for general class metafeatures."""
import pytest

from pymfe.mfe import MFE
from tests.utils import load_xy
import numpy as np

GNAME = "general"


class TestGeneral:
    """TestClass dedicated to test general metafeatures."""

    @pytest.mark.parametrize(
        "dt_id, ft_name, exp_value, precompute",
        [
            ###################
            # Mixed data
            ###################
            (0, "attr_to_inst", 0.08, False),
            (0, "cat_to_num", 1, False),
            (0, "freq_class", [0.50, 0.0], False),
            (0, "inst_to_attr", 12.50, False),
            (0, "nr_attr", 4, False),
            (0, "nr_bin", 0, False),
            (0, "nr_cat", 2, False),
            (0, "nr_class", 2, False),
            (0, "nr_inst", 50, False),
            (0, "nr_num", 2, False),
            (0, "num_to_cat", 1.0, False),
            (0, "attr_to_inst", 0.08, True),
            (0, "cat_to_num", 1, True),
            (0, "freq_class", [0.50, 0.0], True),
            (0, "inst_to_attr", 12.50, True),
            (0, "nr_attr", 4, True),
            (0, "nr_bin", 0, True),
            (0, "nr_cat", 2, True),
            (0, "nr_class", 2, True),
            (0, "nr_inst", 50, True),
            (0, "nr_num", 2, True),
            (0, "num_to_cat", 1.0, True),
            ###################
            # Categorical data
            ###################
            (1, "attr_to_inst", 36 / 3196, False),
            (1, "cat_to_num", np.nan, False),
            (1, "freq_class", [0.5, 0.03141713], False),
            (1, "inst_to_attr", 88.77778, False),
            (1, "nr_attr", 36, False),
            (1, "nr_bin", 35, False),
            (1, "nr_cat", 36, False),
            (1, "nr_class", 2, False),
            (1, "nr_inst", 3196, False),
            (1, "nr_num", 0, False),
            (1, "num_to_cat", 0, False),
            (1, "attr_to_inst", 36 / 3196, True),
            (1, "cat_to_num", np.nan, True),
            (1, "freq_class", [0.5, 0.03141713], True),
            (1, "inst_to_attr", 88.77778, True),
            (1, "nr_attr", 36, True),
            (1, "nr_bin", 35, True),
            (1, "nr_cat", 36, True),
            (1, "nr_class", 2, True),
            (1, "nr_inst", 3196, True),
            (1, "nr_num", 0, True),
            (1, "num_to_cat", 0, True),
            ###################
            # Numerical data
            ###################
            (2, "attr_to_inst", 0.02666667, False),
            (2, "cat_to_num", 0.0, False),
            (2, "freq_class", [0.33333333, 0.0], False),
            (2, "inst_to_attr", 37.50, False),
            (2, "nr_attr", 4, False),
            (2, "nr_bin", 0, False),
            (2, "nr_cat", 0, False),
            (2, "nr_class", 3, False),
            (2, "nr_inst", 150, False),
            (2, "nr_num", 4, False),
            (2, "num_to_cat", np.nan, False),
            (2, "attr_to_inst", 0.02666667, True),
            (2, "cat_to_num", 0.0, True),
            (2, "freq_class", [0.33333333, 0.0], True),
            (2, "inst_to_attr", 37.50, True),
            (2, "nr_attr", 4, True),
            (2, "nr_bin", 0, True),
            (2, "nr_cat", 0, True),
            (2, "nr_class", 3, True),
            (2, "nr_inst", 150, True),
            (2, "nr_num", 4, True),
            (2, "num_to_cat", np.nan, True),
        ],
    )
    def test_ft_methods_general(self, dt_id, ft_name, exp_value, precompute):
        """Function to test each meta-feature belongs to general group."""
        precomp_group = GNAME if precompute else None
        X, y = load_xy(dt_id)
        mfe = MFE(groups=[GNAME], features=[ft_name]).fit(
            X.values, y.values, precomp_groups=precomp_group
        )
        value = mfe.extract()[1]

        if exp_value is np.nan:
            assert value[0] is exp_value

        else:
            assert np.allclose(value, exp_value)

    @pytest.mark.parametrize(
        "dt_id, exp_value, precompute",
        [
            ###################
            # Mixed data
            ###################
            (0, [0.08, 1, 0.50, 12.50, 4, 0, 2, 2, 50, 2, 1.0], False),
            (0, [0.08, 1, 0.50, 12.50, 4, 0, 2, 2, 50, 2, 1.0], True),
            ###################
            # Categorical data
            ###################
            (
                1,
                [36 / 3196, np.nan, 0.5, 88.77778, 36, 35, 36, 2, 3196, 0, 0],
                False,
            ),
            (
                1,
                [36 / 3196, np.nan, 0.5, 88.77778, 36, 35, 36, 2, 3196, 0, 0],
                True,
            ),
            ###################
            # Numerical data
            ###################
            (
                2,
                [
                    0.02666667,
                    0.0,
                    0.33333333,
                    37.50,
                    4,
                    0,
                    0,
                    3,
                    150,
                    4,
                    np.nan,
                ],
                False,
            ),
            (
                2,
                [
                    0.02666667,
                    0.0,
                    0.33333333,
                    37.50,
                    4,
                    0,
                    0,
                    3,
                    150,
                    4,
                    np.nan,
                ],
                True,
            ),
        ],
    )
    def test_integration_general(self, dt_id, exp_value, precompute):
        precomp_group = GNAME if precompute else None
        X, y = load_xy(dt_id)
        mfe = MFE(groups=[GNAME], summary="mean").fit(
            X.values, y.values, precomp_groups=precomp_group
        )
        value = mfe.extract()[1]

        assert np.allclose(value, exp_value, equal_nan=True)
