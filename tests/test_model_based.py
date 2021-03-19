"""Test module for model-based class metafeatures."""
import pytest

from pymfe.mfe import MFE
from tests.utils import load_xy
import numpy as np

GNAME = "model-based"


class TestModelBased:
    """TestClass dedicated to test model-based metafeatures."""

    @pytest.mark.parametrize(
        "dt_id, ft_name, exp_value, precompute",
        [
            ###################
            # Mixed data
            ###################
            (0, "leaves", 13, True),
            (0, "leaves_branch", [4.6153846, 1.4455945], True),
            (0, "leaves_corrob", [0.07692308, 0.058791243], True),
            (0, "leaves_homo", [84.933334, 41.648125], True),
            (0, "leaves_per_class", [0.5, 0.05439285], True),
            (0, "nodes", 12, True),
            (0, "nodes_per_attr", 1.0909090909090908, True),
            (0, "nodes_per_inst", 0.24, True),
            (0, "nodes_per_level", [2.0, 0.8944272], True),
            (0, "nodes_repeated", [3.0, 2.828427], True),
            (0, "tree_depth", [3.84, 1.6753109], True),
            (0, "tree_imbalance", [0.16146065, 0.113601856], True),
            (0, "tree_shape", [0.20192307, 0.1227767], True),
            (0, "var_importance", [0.09090909, 0.1993217], True),
            (0, "leaves", 13, False),
            (0, "leaves_branch", [4.6153846, 1.4455945], False),
            (0, "leaves_corrob", [0.07692308, 0.058791243], False),
            (0, "leaves_homo", [84.933334, 41.648125], False),
            (0, "leaves_per_class", [0.5, 0.05439285], False),
            (0, "nodes", 12, False),
            (0, "nodes_per_attr", 1.0909090909090908, False),
            (0, "nodes_per_inst", 0.24, False),
            (0, "nodes_per_level", [2.0, 0.8944272], False),
            (0, "nodes_repeated", [3.0, 2.828427], False),
            (0, "tree_depth", [3.84, 1.6753109], False),
            (0, "tree_imbalance", [0.16146065, 0.113601856], False),
            (0, "tree_shape", [0.20192307, 0.1227767], False),
            (0, "var_importance", [0.09090909, 0.1993217], False),
            ###################
            # Categorical data
            ###################
            (1, "leaves", 57, True),
            (1, "leaves_branch", [9.140351, 3.136414], True),
            (1, "leaves_corrob", [0.01754386, 0.04135247], True),
            (1, "leaves_homo", [18342.629, 45953.414], True),
            (1, "leaves_per_class", [0.5, 0.11164843], True),
            (1, "nodes", 56, True),
            (1, "nodes_per_attr", 1.4736842105263157, True),
            (1, "nodes_per_inst", 0.017521902377972465, True),
            (1, "nodes_per_level", [3.5, 2.4221203], True),
            (1, "nodes_repeated", [1.6969697, 0.88334763], True),
            (1, "tree_depth", [8.230088, 3.305863], True),
            (1, "tree_imbalance", [0.05483275, 0.092559], True),
            (1, "tree_shape", [0.052245557, 0.09386974], True),
            (1, "var_importance", [0.02631579, 0.06340529], True),
            (1, "leaves", 57, False),
            (1, "leaves_branch", [9.140351, 3.136414], False),
            (1, "leaves_corrob", [0.01754386, 0.04135247], False),
            (1, "leaves_homo", [18342.629, 45953.414], False),
            (1, "leaves_per_class", [0.5, 0.11164843], False),
            (1, "nodes", 56, False),
            (1, "nodes_per_attr", 1.4736842105263157, False),
            (1, "nodes_per_inst", 0.017521902377972465, False),
            (1, "nodes_per_level", [3.5, 2.4221203], False),
            (1, "nodes_repeated", [1.6969697, 0.88334763], False),
            (1, "tree_depth", [8.230088, 3.305863], False),
            (1, "tree_imbalance", [0.05483275, 0.092559], False),
            (1, "tree_shape", [0.052245557, 0.09386974], False),
            (1, "var_importance", [0.02631579, 0.06340529], False),
            ###################
            # Numerical data
            ###################
            (2, "leaves", 9, True),
            (2, "leaves_branch", [3.7777777, 1.2018504], True),
            (2, "leaves_corrob", [0.11111111, 0.15051763], True),
            (2, "leaves_homo", [37.466667, 13.142298], True),
            (2, "leaves_per_class", [0.33333334, 0.22222224], True),
            (2, "nodes", 8, True),
            (2, "nodes_per_attr", 2.0, True),
            (2, "nodes_per_inst", 0.05333333333333334, True),
            (2, "nodes_per_level", [1.6, 0.8944272], True),
            (2, "nodes_repeated", [2.0, 1.1547005], True),
            (2, "tree_depth", [3.0588236, 1.4348601], True),
            (2, "tree_imbalance", [0.19491705, 0.1330071], True),
            (2, "tree_shape", [0.27083334, 0.107119605], True),
            (2, "var_importance", [0.24999999, 0.27823895], True),
            (2, "leaves", 9, False),
            (2, "leaves_branch", [3.7777777, 1.2018504], False),
            (2, "leaves_corrob", [0.11111111, 0.15051763], False),
            (2, "leaves_homo", [37.466667, 13.142298], False),
            (2, "leaves_per_class", [0.33333334, 0.22222224], False),
            (2, "nodes", 8, False),
            (2, "nodes_per_attr", 2.0, False),
            (2, "nodes_per_inst", 0.05333333333333334, False),
            (2, "nodes_per_level", [1.6, 0.8944272], False),
            (2, "nodes_repeated", [2.0, 1.1547005], False),
            (2, "tree_depth", [3.0588236, 1.4348601], False),
            (2, "tree_imbalance", [0.19491705, 0.1330071], False),
            (2, "tree_shape", [0.27083334, 0.107119605], False),
            (2, "var_importance", [0.24999999, 0.27823895], False),
        ],
    )
    def test_ft_methods_model_based_01(
        self, dt_id, ft_name, exp_value, precompute
    ):
        """Function to test each meta-feature belongs to model-based group."""
        precomp_group = GNAME if precompute else None

        X, y = load_xy(dt_id)
        mfe = MFE(groups=[GNAME], features=[ft_name], random_state=1234)

        mfe.fit(X.values, y.values, precomp_groups=precomp_group)

        value = mfe.extract()[1]

        if exp_value is np.nan:
            assert value[0] is exp_value

        else:
            assert np.allclose(value, exp_value)

    @pytest.mark.parametrize(
        "dt_id, ft_name, exp_value, precompute",
        [
            ###################
            # Mixed data
            ###################
            (0, "leaves", 7, True),
            (0, "leaves_branch", [3.7142856, 1.7043363], True),
            (0, "leaves_corrob", [0.14285713, 0.06575568], True),
            (0, "leaves_homo", [32.266666, 15.709021], True),
            (0, "leaves_per_class", [0.5, 0.30304578], True),
            (0, "nodes", 6, True),
            (0, "nodes_per_attr", 0.5454545454545454, True),
            (0, "nodes_per_inst", 0.12, True),
            (0, "nodes_per_level", [1.2, 0.4472136], True),
            (0, "nodes_repeated", [3.0, 1.4142135], True),
            (0, "tree_depth", [3.0769231, 1.7541162], True),
            (0, "tree_imbalance", [0.19825712, 0.11291388], True),
            (0, "tree_shape", [0.2857143, 0.16675964], True),
            (0, "var_importance", [0.09090909, 0.2417293], True),
            (0, "leaves", 7, False),
            (0, "leaves_branch", [3.7142856, 1.7043363], False),
            (0, "leaves_corrob", [0.14285713, 0.06575568], False),
            (0, "leaves_homo", [32.266666, 15.709021], False),
            (0, "leaves_per_class", [0.5, 0.30304578], False),
            (0, "nodes", 6, False),
            (0, "nodes_per_attr", 0.5454545454545454, False),
            (0, "nodes_per_inst", 0.12, False),
            (0, "nodes_per_level", [1.2, 0.4472136], False),
            (0, "nodes_repeated", [3.0, 1.4142135], False),
            (0, "tree_depth", [3.0769231, 1.7541162], False),
            (0, "tree_imbalance", [0.19825712, 0.11291388], False),
            (0, "tree_shape", [0.2857143, 0.16675964], False),
            (0, "var_importance", [0.09090909, 0.2417293], False),
            ###################
            # Categorical data
            ###################
            (1, "leaves", 10, True),
            (1, "leaves_branch", [4.3, 1.4944341], True),
            (1, "leaves_corrob", [0.1, 0.08727827], True),
            (1, "leaves_homo", [55.2, 18.552029], True),
            (1, "leaves_per_class", [0.5, 0.2828427], True),
            (1, "nodes", 9, True),
            (1, "nodes_per_attr", 0.23684210526315788, True),
            (1, "nodes_per_inst", 0.002816020025031289, True),
            (1, "nodes_per_level", [1.8, 1.3038405], True),
            (1, "nodes_repeated", [1.125, 0.35355338], True),
            (1, "tree_depth", [3.5789473, 1.6437014], True),
            (1, "tree_imbalance", [0.25800052, 0.0827512], True),
            (1, "tree_shape", [0.225, 0.14493772], True),
            (1, "var_importance", [0.02631579, 0.07277515], True),
            (1, "leaves", 10, False),
            (1, "leaves_branch", [4.3, 1.4944341], False),
            (1, "leaves_corrob", [0.1, 0.08727827], False),
            (1, "leaves_homo", [55.2, 18.552029], False),
            (1, "leaves_per_class", [0.5, 0.2828427], False),
            (1, "nodes", 9, False),
            (1, "nodes_per_attr", 0.23684210526315788, False),
            (1, "nodes_per_inst", 0.002816020025031289, False),
            (1, "nodes_per_level", [1.8, 1.3038405], False),
            (1, "nodes_repeated", [1.125, 0.35355338], False),
            (1, "tree_depth", [3.5789473, 1.6437014], False),
            (1, "tree_imbalance", [0.25800052, 0.0827512], False),
            (1, "tree_shape", [0.225, 0.14493772], False),
            (1, "var_importance", [0.02631579, 0.07277515], False),
            ###################
            # Numerical data
            ###################
            (2, "leaves", 6, True),
            (2, "leaves_branch", [3.0, 1.0954452], True),
            (2, "leaves_corrob", [0.16666667, 0.15927614], True),
            (2, "leaves_homo", [18.0, 4.8989797], True),
            (2, "leaves_per_class", [0.33333334, 0.28867516], True),
            (2, "nodes", 5, True),
            (2, "nodes_per_attr", 1.25, True),
            (2, "nodes_per_inst", 0.03333333333333333, True),
            (2, "nodes_per_level", [1.25, 0.5], True),
            (2, "nodes_repeated", [2.5, 0.70710677], True),
            (2, "tree_depth", [2.3636363, 1.2862914], True),
            (2, "tree_imbalance", [0.2524478, 0.1236233], True),
            (2, "tree_shape", [0.35416666, 0.094096586], True),
            (2, "var_importance", [0.25, 0.31985083], True),
            (2, "leaves", 6, False),
            (2, "leaves_branch", [3.0, 1.0954452], False),
            (2, "leaves_corrob", [0.16666667, 0.15927614], False),
            (2, "leaves_homo", [18.0, 4.8989797], False),
            (2, "leaves_per_class", [0.33333334, 0.28867516], False),
            (2, "nodes", 5, False),
            (2, "nodes_per_attr", 1.25, False),
            (2, "nodes_per_inst", 0.03333333333333333, False),
            (2, "nodes_per_level", [1.25, 0.5], False),
            (2, "nodes_repeated", [2.5, 0.70710677], False),
            (2, "tree_depth", [2.3636363, 1.2862914], False),
            (2, "tree_imbalance", [0.2524478, 0.1236233], False),
            (2, "tree_shape", [0.35416666, 0.094096586], False),
            (2, "var_importance", [0.25, 0.31985083], False),
        ],
    )
    def test_ft_methods_model_based_02(
        self, dt_id, ft_name, exp_value, precompute
    ):
        """Function to test each meta-feature belongs to model-based group."""
        precomp_group = GNAME if precompute else None

        X, y = load_xy(dt_id)
        mfe = MFE(
            groups=[GNAME],
            features=[ft_name],
            hypparam_model_dt={
                "max_depth": 5,
                "min_samples_split": 10,
                "criterion": "entropy",
            },
            random_state=1234,
        )

        mfe.fit(X.values, y.values, precomp_groups=precomp_group)

        if precomp_group is None:
            # Note: the precomputation of 'model-based' group is always
            # forced due to the need of the 'dt_model' value
            mfe._precomp_args_ft = {
                "dt_model": mfe._precomp_args_ft.get("dt_model")
            }

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
            (
                0,
                [
                    13,
                    4.6153846,
                    0.07692308,
                    84.933334,
                    0.5,
                    12,
                    1.0909090909090908,
                    0.24,
                    2.0,
                    3.0,
                    3.84,
                    0.16146065,
                    0.20192307,
                    0.09090909,
                ],
                False,
            ),
            (
                0,
                [
                    13,
                    4.6153846,
                    0.07692308,
                    84.933334,
                    0.5,
                    12,
                    1.0909090909090908,
                    0.24,
                    2.0,
                    3.0,
                    3.84,
                    0.16146065,
                    0.20192307,
                    0.09090909,
                ],
                True,
            ),
            ###################
            # Numerical data
            ###################
            (
                2,
                [
                    9,
                    3.7777777,
                    0.11111111,
                    37.466667,
                    0.33333334,
                    8,
                    2.0,
                    0.05333333333333334,
                    1.6,
                    2.0,
                    3.0588236,
                    0.19491705,
                    0.27083334,
                    0.24999999,
                ],
                False,
            ),
            (
                2,
                [
                    9,
                    3.7777777,
                    0.11111111,
                    37.466667,
                    0.33333334,
                    8,
                    2.0,
                    0.05333333333333334,
                    1.6,
                    2.0,
                    3.0588236,
                    0.19491705,
                    0.27083334,
                    0.24999999,
                ],
                True,
            ),
        ],
    )
    def test_integration_model_based(self, dt_id, exp_value, precompute):
        """Function to test all model-based meta-features."""
        precomp_group = GNAME if precompute else None

        X, y = load_xy(dt_id)
        mfe = MFE(groups=[GNAME], summary="mean", random_state=1234)

        mfe.fit(X.values, y.values, precomp_groups=precomp_group)

        value = mfe.extract()[1]

        assert np.allclose(value, exp_value, equal_nan=True)
