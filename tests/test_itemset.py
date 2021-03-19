"""Test module for itemset metafeatures."""
import pytest
import numpy as np

from pymfe.mfe import MFE
from tests.utils import load_xy
from pymfe.itemset import MFEItemset

GNAME = "itemset"


class TestItemset:
    """TestClass dedicated to test itemset metafeatures."""

    @pytest.mark.parametrize(
        "dt_id, ft_name, exp_value, precompute",
        [
            ###################
            # Mixed data
            ###################
            (0, "one_itemset", [0.24999999, 0.0669328], True),
            (0, "two_itemset", [0.38297877, 0.10911008], True),
            (0, "one_itemset", [0.24999999, 0.0669328], False),
            (0, "two_itemset", [0.38297877, 0.10911008], False),
            ###################
            # Categorical data
            ###################
            (1, "one_itemset", [0.49315068, 0.34882316], True),
            (1, "two_itemset", [0.5, 0.24335141], True),
            (1, "one_itemset", [0.49315068, 0.34882316], False),
            (1, "two_itemset", [0.5, 0.24335141], False),
            ###################
            # Numerical data
            ###################
            (2, "one_itemset", [0.2, 0.049322903], True),
            (2, "two_itemset", [0.32, 0.084694475], True),
            (2, "one_itemset", [0.2, 0.049322903], False),
            (2, "two_itemset", [0.32, 0.084694475], False),
        ],
    )
    def test_ft_methods_itemset(self, dt_id, ft_name, exp_value, precompute):
        """Function to test each meta-feature belongs to itemset group."""
        precomp_group = GNAME if precompute else None

        X, y = load_xy(dt_id)
        mfe = MFE(groups=[GNAME], features=[ft_name], random_state=1234)

        mfe.fit(X.values, y.values, precomp_groups=precomp_group)

        value = mfe.extract()[1]

        if exp_value is np.nan:
            assert value[0] is exp_value
        else:
            assert np.allclose(value, exp_value, equal_nan=True)

    def test_itemset_using_author_dataset(self):
        """In this test we use the toy dataset and results used by the authors"
        paper.
        """
        C = np.array(
            [
                [0, 2, 3],
                [2, 5, 0],
                [1, 4, 1],
                [0, 2, 2],
                [3, 3, 3],
                [3, 2, 3],
                [0, 2, 0],
                [1, 3, 1],
                [2, 4, 3],
                [1, 5, 2],
            ]
        )

        value = MFEItemset.ft_one_itemset(C=C)
        exp_value = [
            0.3,
            0.3,
            0.2,
            0.2,
            0.4,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.4,
        ]

        assert np.allclose(value, exp_value, equal_nan=True)

        value = MFEItemset.ft_two_itemset(C=C)
        exp_value = [0.1, 0.5, 0.5]

        assert np.allclose(value[[0, 1, 2]], exp_value, equal_nan=True)

        exp_value = [0.2, 0.6]
        assert np.allclose(value[[-2, -1]], exp_value, equal_nan=True)

    @pytest.mark.parametrize(
        "dt_id, exp_value, precompute",
        [
            ###################
            # Mixed data
            ###################
            (0, [0.24999999, 0.38297877], False),
            (0, [0.24999999, 0.38297877], True),
            ###################
            # Numerical data
            ###################
            (2, [0.2, 0.32], False),
            (2, [0.2, 0.32], True),
        ],
    )
    def test_integration_itemset(self, dt_id, exp_value, precompute):
        """Function to test all itemset meta-features."""
        precomp_group = GNAME if precompute else None

        X, y = load_xy(dt_id)
        mfe = MFE(groups=[GNAME], summary="mean", random_state=1234)

        mfe.fit(X.values, y.values, precomp_groups=precomp_group)

        value = mfe.extract()[1]

        assert np.allclose(value, exp_value, equal_nan=True)
