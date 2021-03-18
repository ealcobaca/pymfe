"""Test module for information theory class metafeatures."""
import pytest

from pymfe.mfe import MFE
from tests.utils import load_xy
import numpy as np

GNAME = "info-theory"


class TestInfoTheo:
    """TestClass dedicated to test information theory metafeatures."""

    @pytest.mark.parametrize(
        "dt_id, ft_name, exp_value, precompute",
        [
            ###################
            # Mixed data
            ###################
            (0, "attr_conc", [2.098425e-01, 2.139363e-01], False),
            (0, "attr_ent", [1.953155e00, 4.258227e-01], False),
            (0, "class_conc", [7.803121e-03, 1.075977e-02], False),
            (0, "class_ent", 1, False),
            (0, "eq_num_attr", 8.894722e01, False),
            (0, "joint_ent", [2.941912e00, 4.388859e-01], False),
            (0, "mut_inf", [1.124262e-02, 1.548654e-02], False),
            (0, "ns_ratio", 1.727277e02, False),
            (0, "attr_conc", [2.098425e-01, 2.139363e-01], True),
            (0, "attr_ent", [1.953155e00, 4.258227e-01], True),
            (0, "class_conc", [7.803121e-03, 1.075977e-02], True),
            (0, "class_ent", 1, True),
            (0, "eq_num_attr", 8.894722e01, True),
            (0, "joint_ent", [2.941912e00, 4.388859e-01], True),
            (0, "mut_inf", [1.124262e-02, 1.548654e-02], True),
            (0, "ns_ratio", 1.727277e02, True),
            ###################
            # Categorical data
            ###################
            (1, "attr_conc", [0.017922703, 0.057748884], False),
            (1, "attr_ent", [0.59014829, 0.33852165], False),
            (1, "class_conc", [0.02313025, 0.04485300], False),
            (1, "class_ent", 0.99857554, False),
            (1, "eq_num_attr", 52.14040170, False),
            (1, "joint_ent", [1.56957216, 0.33197232], False),
            (1, "mut_inf", [0.01915167, 0.03918710], False),
            (1, "ns_ratio", 29.81446298, False),
            (1, "attr_conc", [0.017922703, 0.057748884], True),
            (1, "attr_ent", [0.59014829, 0.33852165], True),
            (1, "class_conc", [0.02313025, 0.04485300], True),
            (1, "class_ent", 0.99857554, True),
            (1, "eq_num_attr", 52.14040170, True),
            (1, "joint_ent", [1.56957216, 0.33197232], True),
            (1, "mut_inf", [0.01915167, 0.03918710], True),
            (1, "ns_ratio", 29.81446298, True),
            ###################
            # Numerical data
            ###################
            (2, "attr_conc", [0.20922253, 0.11995021], False),
            (2, "attr_ent", [2.27901045, 0.05742642], False),
            (2, "class_conc", [0.27232600, 0.14258949], False),
            (2, "class_ent", 1.58496250, False),
            (2, "eq_num_attr", 1.88240501, False),
            (2, "joint_ent", [3.02198491, 0.38738119], False),
            (2, "mut_inf", [0.84198804, 0.42518056], False),
            (2, "ns_ratio", 1.70670169, False),
            (2, "attr_conc", [0.20922253, 0.11995021], True),
            (2, "attr_ent", [2.27901045, 0.05742642], True),
            (2, "class_conc", [0.27232600, 0.14258949], True),
            (2, "class_ent", 1.58496250, True),
            (2, "eq_num_attr", 1.88240501, True),
            (2, "joint_ent", [3.02198491, 0.38738119], True),
            (2, "mut_inf", [0.84198804, 0.42518056], True),
            (2, "ns_ratio", 1.70670169, True),
        ],
    )
    def test_ft_methods_infotheo(self, dt_id, ft_name, exp_value, precompute):
        """Function to test each meta-feature belongs to info-theory group."""
        precomp_group = GNAME if precompute else None
        X, y = load_xy(dt_id)
        mfe = MFE(groups=[GNAME], features=[ft_name], random_state=1234).fit(
            X.values, y.values, precomp_groups=precomp_group
        )
        value = mfe.extract()[1]

        if exp_value is np.nan:
            assert value[0] is exp_value

        else:
            assert np.allclose(
                value, exp_value, atol=0.001, rtol=0.05, equal_nan=True
            )

    @pytest.mark.parametrize(
        "dt_id, exp_value, precompute",
        [
            ###################
            # Mixed data
            ###################
            (
                0,
                [
                    2.098425e-01,
                    1.953155e00,
                    7.803121e-03,
                    1,
                    8.894722e01,
                    2.941912e00,
                    1.124262e-02,
                    1.727277e02,
                ],
                False,
            ),
            (
                0,
                [
                    2.098425e-01,
                    1.953155e00,
                    7.803121e-03,
                    1,
                    8.894722e01,
                    2.941912e00,
                    1.124262e-02,
                    1.727277e02,
                ],
                True,
            ),
            ###################
            # Numerical data
            ###################
            (
                2,
                [
                    0.20922253,
                    2.27901045,
                    0.27232600,
                    1.58496250,
                    1.88240501,
                    3.02198491,
                    0.84198804,
                    1.70670169,
                ],
                False,
            ),
            (
                2,
                [
                    0.20922253,
                    2.27901045,
                    0.27232600,
                    1.58496250,
                    1.88240501,
                    3.02198491,
                    0.84198804,
                    1.70670169,
                ],
                True,
            ),
        ],
    )
    def test_integration_infotheo(self, dt_id, exp_value, precompute):
        """Function to test all info-theory meta-features."""
        precomp_group = GNAME if precompute else None
        X, y = load_xy(dt_id)
        mfe = MFE(groups=[GNAME], summary="mean").fit(
            X.values, y.values, precomp_groups=precomp_group
        )
        value = mfe.extract()[1]

        np.allclose(value, exp_value, atol=0.001, rtol=0.05, equal_nan=True)

    def test_threshold_attr_conc(self):
        X, y = load_xy(1)
        mfe = MFE(features="attr_conc", random_state=1234).fit(
            X.values, y.values, precomp_groups=False
        )

        value = mfe.extract(attr_conc={"max_attr_num": 25})[1]

        assert np.allclose(value, [0.01682327, 0.04715381], rtol=0.2)
