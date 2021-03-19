"""Test module for MFE data scaling."""
import pytest

import numpy as np

from pymfe.mfe import MFE
from tests.utils import load_xy

GNAME = "data-scaling"


class TestDataScaling:
    """TestClass dedicated to test MFE scaling."""

    @pytest.mark.parametrize(
        "dt_id, scaler, exp_mean, exp_var, exp_min, exp_max",
        [
            (
                0,
                "min-max",
                [0.10972805, 0.18916243],
                [0.04092028, 0.07842329],
                [0, 0],
                [1, 1],
            ),
            (
                2,
                "min-max",
                [0.4287037, 0.43916667, 0.46757062, 0.45777778],
                [0.05255573, 0.03242199, 0.08883726, 0.10043951],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ),
            (
                0,
                "robust",
                [0.82562938, 1.33437416],
                [3.45558848, 8.99152336],
                [-0.18271619, -0.69111064],
                [9.0067807, 10.01653805],
            ),
            (
                2,
                "robust",
                [0.03333333, 0.108, -0.16895238, -0.06755556],
                [0.4030309, 0.74700267, 0.25244285, 0.25712514],
                [-1.15384615, -2.0, -0.95714286, -0.8],
                [1.61538462, 2.8, 0.72857143, 0.8],
            ),
            (
                0,
                "standard",
                [0, 0],
                [1, 1],
                [-0.54243585, -0.67547977],
                [4.40102076, 2.89541848],
            ),
            (
                2,
                "standard",
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [-1.87002413, -2.43898725, -1.56873522, -1.4444497],
                [2.4920192, 3.11468391, 1.78634131, 1.71090158],
            ),
        ],
    )
    def test_output_lengths_2(
        self, dt_id, scaler, exp_mean, exp_var, exp_min, exp_max
    ):
        X, y = load_xy(dt_id)
        model = MFE().fit(
            X=X.values, y=y.values, rescale=scaler, transform_cat=None
        )

        numeric_data = model._custom_args_ft["N"]

        assert (
            np.allclose(numeric_data.mean(axis=0), exp_mean)
            and np.allclose(numeric_data.var(axis=0), exp_var)
            and np.allclose(numeric_data.min(axis=0), exp_min)
            and np.allclose(numeric_data.max(axis=0), exp_max)
        )

    def test_scaling_error_1(self):
        with pytest.raises(ValueError):
            X, y = load_xy(0)
            MFE().fit(
                X=X.values, y=y.values, rescale="invalid", transform_cat=False
            )

    def test_scaling_error_2(self):
        with pytest.raises(TypeError):
            X, y = load_xy(0)
            MFE().fit(
                X=X.values,
                y=y.values,
                rescale=["invalid"],
                transform_cat=False,
            )
