"""Test module for system testing."""
import pytest

from pymfe.mfe import MFE
from tests.utils import load_xy
import numpy as np


class TestSystem:
    """TestClass for system testing.

    The purpose is the check if, while extracting all available metafeatures,
    the computation of any feature does not interfere in the computation of
    metafeatures from other groups.
    """

    @pytest.mark.parametrize(
        "dt_id, precompute, supervised",
        [
            (0, False, True),
            (0, True, True),
            (2, False, True),
            (2, True, True),
            (0, False, False),
            (0, True, False),
            (2, False, False),
            (2, True, False),
        ],
    )
    def test_system_testing(self, dt_id, precompute, supervised):
        precomp_group = "all" if precompute else None
        X, y = load_xy(dt_id)

        mtf_groups = (
            "landmarking",
            "general",
            "statistical",
            "model-based",
            "info-theory",
            "relative",
            "clustering",
            "complexity",
            "itemset",
            "concept",
        )

        def extract_mtf_by_group():
            all_mtf_names = []
            all_mtf_vals = []

            for cur_group in mtf_groups:
                cur_precomp_group = cur_group if precompute else None

                mfe = MFE(
                    groups=cur_group, summary="mean", random_state=1234
                ).fit(
                    X.values,
                    y.values if supervised else None,
                    precomp_groups=cur_precomp_group,
                )

                cur_names, cur_vals = mfe.extract()

                all_mtf_names += cur_names
                all_mtf_vals += cur_vals

            _, all_mtf_vals = zip(
                *sorted(
                    zip(all_mtf_names, all_mtf_vals), key=lambda item: item[0]
                )
            )

            return all_mtf_vals

        def extract_all_mtf():
            mfe = MFE(
                groups=mtf_groups, summary="mean", random_state=1234
            ).fit(
                X.values,
                y.values if supervised else None,
                precomp_groups=precomp_group,
            )

            all_mtf_vals = mfe.extract()[1]

            return all_mtf_vals

        assert np.allclose(
            extract_all_mtf(),
            extract_mtf_by_group(),
            equal_nan=True,
            rtol=0.05,
            atol=0.001,
        )
