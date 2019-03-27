"""Test module for ModelBased class metafeatures."""
import pytest

from pymfe.mfe import MFE
from tests.utils import load_xy
import numpy as np

GNAME = "model_based"


class TestModelBased():
    """TestClass dedicated to test ModelBased metafeatures."""

    @pytest.mark.parametrize(
        "dt_id, ft_name, exp_value, precompute",
        [
            ###################
            # Mixed data
            ###################
            (0, 'leaves', 13, True),
            (0, 'leaves_branch', [4.6153846, 1.4455945], True),
            (0, 'leaves_corrob', [0.07692308, 0.058791243], True),
            (0, 'leaves_homo', [84.933334, 41.648125], True),
            (0, 'leaves_per_class', [0.5, 0.05439285], True),
            (0, 'nodes', 12, True),
            (0, 'nodes_per_attr', 1.0909090909090908, True),
            (0, 'nodes_per_inst', 0.24, True),
            (0, 'nodes_per_level', [2.0, 0.8944272], True),
            (0, 'nodes_repeated', [1.6666666, 1.1547005], True),
            (0, 'tree_depth', [3.84, 1.6753109], True),
            (0, 'tree_imbalance', [0.16146065, 0.113601856], True),
            (0, 'tree_shape', [0.20192307, 0.1227767], True),
            (0, 'var_importance', [0.09090909, 0.1993217], True),
            (0, 'leaves', 13, False),
            (0, 'leaves_branch', [4.6153846, 1.4455945], False),
            (0, 'leaves_corrob', [0.07692308, 0.058791243], False),
            (0, 'leaves_homo', [84.933334, 41.648125], False),
            (0, 'leaves_per_class', [0.5, 0.05439285], False),
            (0, 'nodes', 12, False),
            (0, 'nodes_per_attr', 1.0909090909090908, False),
            (0, 'nodes_per_inst', 0.24, False),
            (0, 'nodes_per_level', [2.0, 0.8944272], False),
            (0, 'nodes_repeated', [1.6666666, 1.1547005], False),
            (0, 'tree_depth', [3.84, 1.6753109], False),
            (0, 'tree_imbalance', [0.16146065, 0.113601856], False),
            (0, 'tree_shape', [0.20192307, 0.1227767], False),
            (0, 'var_importance', [0.09090909, 0.1993217], False),
            ###################
            # Categorical data
            ###################
            (1, 'leaves', 57, True),
            (1, 'leaves_branch', [9.140351, 3.136414], True),
            (1, 'leaves_corrob', [0.01754386, 0.04135247], True),
            (1, 'leaves_homo', [18342.629, 45953.414], True),
            (1, 'leaves_per_class', [0.5, 0.11164843], True),
            (1, 'nodes', 56, True),
            (1, 'nodes_per_attr', 1.4736842105263157, True),
            (1, 'nodes_per_inst', 0.017521902377972465, True),
            (1, 'nodes_per_level', [3.5, 2.4221203], True),
            (1, 'nodes_repeated', [1.65625, 0.86544317], True),
            (1, 'tree_depth', [8.230088, 3.305863], True),
            (1, 'tree_imbalance', [0.05483275, 0.092559], True),
            (1, 'tree_shape', [0.052245557, 0.09386974], True),
            (1, 'var_importance', [0.02631579, 0.06340529], True),
            (1, 'leaves', 57, False),
            (1, 'leaves_branch', [9.140351, 3.136414], False),
            (1, 'leaves_corrob', [0.01754386, 0.04135247], False),
            (1, 'leaves_homo', [18342.629, 45953.414], False),
            (1, 'leaves_per_class', [0.5, 0.11164843], False),
            (1, 'nodes', 56, False),
            (1, 'nodes_per_attr', 1.4736842105263157, False),
            (1, 'nodes_per_inst', 0.017521902377972465, False),
            (1, 'nodes_per_level', [3.5, 2.4221203], False),
            (1, 'nodes_repeated', [1.65625, 0.86544317], False),
            (1, 'tree_depth', [8.230088, 3.305863], False),
            (1, 'tree_imbalance', [0.05483275, 0.092559], False),
            (1, 'tree_shape', [0.052245557, 0.09386974], False),
            (1, 'var_importance', [0.02631579, 0.06340529], False),
            ###################
            # Numerical data
            ###################
            (2, 'leaves', 9, True),
            (2, 'leaves_branch', [3.7777777, 1.2018504], True),
            (2, 'leaves_corrob', [0.11111111, 0.15051763], True),
            (2, 'leaves_homo', [37.466667, 13.142298], True),
            (2, 'leaves_per_class', [0.33333334, 0.22222224], True),
            (2, 'nodes', 8, True),
            (2, 'nodes_per_attr', 2.0, True),
            (2, 'nodes_per_inst', 0.05333333333333334, True),
            (2, 'nodes_per_level', [1.6, 0.8944272], True),
            (2, 'nodes_repeated', [2.3333333, 1.1547005], True),
            (2, 'tree_depth', [3.0588236, 1.4348601], True),
            (2, 'tree_imbalance', [0.19491705, 0.1330071], True),
            (2, 'tree_shape', [0.27083334, 0.107119605], True),
            (2, 'var_importance', [0.24999999, 0.27823895], True),
            (2, 'leaves', 9, False),
            (2, 'leaves_branch', [3.7777777, 1.2018504], False),
            (2, 'leaves_corrob', [0.11111111, 0.15051763], False),
            (2, 'leaves_homo', [37.466667, 13.142298], False),
            (2, 'leaves_per_class', [0.33333334, 0.22222224], False),
            (2, 'nodes', 8, False),
            (2, 'nodes_per_attr', 2.0, False),
            (2, 'nodes_per_inst', 0.05333333333333334, False),
            (2, 'nodes_per_level', [1.6, 0.8944272], False),
            (2, 'nodes_repeated', [2.3333333, 1.1547005], False),
            (2, 'tree_depth', [3.0588236, 1.4348601], False),
            (2, 'tree_imbalance', [0.19491705, 0.1330071], False),
            (2, 'tree_shape', [0.27083334, 0.107119605], False),
            (2, 'var_importance', [0.24999999, 0.27823895], False),
        ])
    def test_ft_methods_model_based(self, dt_id, ft_name, exp_value,
                                    precompute):
        """Function to test each meta-feature belongs to model_based group.
        """
        precomp_group = "model-based" if precompute else None

        X, y = load_xy(dt_id)
        mfe = MFE(
            groups=["model-based"], features=[ft_name], random_state=1234)

        mfe.fit(X.values, y.values, precomp_groups=precomp_group)

        value = mfe.extract()[1]

        if exp_value is np.nan:
            assert value[0] is exp_value

        else:
            assert np.allclose(value, exp_value)
