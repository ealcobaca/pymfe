"""Test module for Relative Landmarking class metafeatures."""
import pytest

from pymfe.mfe import MFE
from tests.utils import load_xy
import numpy as np

GNAME = "landmarking"


class TestRelativeLandmarking():

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
