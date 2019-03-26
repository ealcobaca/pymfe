"""Test module for ModelBased class metafeatures."""
import pytest

from pymfe.mfe import MFE
from tests.utils import load_xy
import numpy as np

GNAME = "model_based"


class TestModelBased():
    """TestClass dedicated to test ModelBased metafeatures."""
    @pytest.mark.parametrize("dt_id, ft_name, exp_value", [
        ###################
        # Mixed data
        ###################
        (0, 'leaves', 13),
        (0, 'leaves_branch', [4.6153846, 1.4455945]),
        (0, 'leaves_corrob', [0.07692308, 0.058791243]),
        (0, 'leaves_homo', [84.933334, 41.648125]),
        (0, 'leaves_per_class', [0.5, 0.05439285]),
        (0, 'nodes', 12),
        (0, 'nodes_per_attr', 1.0909090909090908),
        (0, 'nodes_per_inst', 0.24),
        (0, 'nodes_per_level', [2.0, 0.8944272]),
        (0, 'nodes_repeated', [1.6666666, 1.1547005]),
        (0, 'tree_depth', [3.84, 1.6753109]),
        (0, 'tree_imbalance', [0.16146065, 0.113601856]),
        (0, 'tree_shape', [0.20192307, 0.1227767]),
        (0, 'var_importance', [0.09090909, 0.1993217]),
        ###################
        # Categorical data
        ###################
        (1, 'leaves', 57),
        (1, 'leaves_branch', [9.140351, 3.136414]),
        (1, 'leaves_corrob', [0.01754386, 0.04135247]),
        (1, 'leaves_homo', [18342.629, 45953.414]),
        (1, 'leaves_per_class', [0.5, 0.11164843]),
        (1, 'nodes', 56),
        (1, 'nodes_per_attr', 1.4736842105263157),
        (1, 'nodes_per_inst', 0.017521902377972465),
        (1, 'nodes_per_level', [3.5, 2.4221203]),
        (1, 'nodes_repeated', [1.65625, 0.86544317]),
        (1, 'tree_depth', [8.230088, 3.305863]),
        (1, 'tree_imbalance', [0.05483275, 0.092559]),
        (1, 'tree_shape', [0.052245557, 0.09386974]),
        (1, 'var_importance', [0.02631579, 0.06340529]),
        ###################
        # Numerical data
        ###################
        (2, 'leaves', 9),
        (2, 'leaves_branch', [3.7777777, 1.2018504]),
        (2, 'leaves_corrob', [0.11111111, 0.15051763]),
        (2, 'leaves_homo', [37.466667, 13.142298]),
        (2, 'leaves_per_class', [0.33333334, 0.22222224]),
        (2, 'nodes', 8),
        (2, 'nodes_per_attr', 2.0),
        (2, 'nodes_per_inst', 0.05333333333333334),
        (2, 'nodes_per_level', [1.6, 0.8944272]),
        (2, 'nodes_repeated', [2.3333333, 1.1547005,]),
        (2, 'tree_depth', [3.0588236, 1.4348601]),
        (2, 'tree_imbalance', [0.19491705, 0.1330071]),
        (2, 'tree_shape', [0.27083334, 0.107119605]),
        (2, 'var_importance', [0.24999999, 0.27823895]),
    ])
    def test_ft_methods_model_based(self, dt_id, ft_name, exp_value):
        """Function to test each meta-feature belongs to model_based group.
        """
        X, y = load_xy(dt_id)
        mfe = MFE(groups=["model-based"],
                  features=[ft_name], random_state=1234).fit(X.values, y.values)
        value = mfe.extract()[1]

        if exp_value is np.nan:
            assert value[0] is exp_value

        else:
            assert np.allclose(value, exp_value)
