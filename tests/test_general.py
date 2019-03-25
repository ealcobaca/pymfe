"""Test module for General class metafeatures."""
import pytest

from pymfe.mfe import MFE
import utils
import numpy as np

GNAME = "general"


class TestGeneral():
    """TestClass dedicated to test General metafeatures."""
    @pytest.mark.parametrize("dt_id, ft_name, exp_value", [
        ###################
        # Mixed data
        ###################
        (0, 'attr_to_inst', 0.08),
        # (0, 'cat_to_num', 1),
        (0, 'freq_class', [0.50, 0.0]),
        (0, 'inst_to_attr', 12.50),
        (0, 'nr_attr', 4),
        (0, 'nr_bin', 0),
        # (0, 'nr_cat', 2),
        (0, 'nr_class', 2),
        (0, 'nr_inst', 50),
        # (0, 'nr_num', 11),
        # (0, 'num_to_cat', 2.75)
        ###################
        # Categorical data
        ###################
        (1, 'attr_to_inst', 36/3196),
        # (1, 'cat_to_num', np.nan),
        (1, 'freq_class', [0.5, 0.03141713]),
        (1, 'inst_to_attr', 88.77778),
        (1, 'nr_attr', 36),
        (1, 'nr_bin', 35),
        (1, 'nr_cat', 36),
        (1, 'nr_class', 2),
        (1, 'nr_inst', 3196),
        # (1, 'nr_num', 0),
        # (1, 'num_to_cat', 0)
        ###################
        # Categorical data
        ###################
        (2, 'attr_to_inst', 0.02666667),
        # (0, 'cat_to_num', 0.0),
        (2, 'freq_class', [0.33333333, 0.0]),
        (2, 'inst_to_attr', 37.50),
        (2, 'nr_attr', 4),
        (2, 'nr_bin', 0),
        # (0, 'nr_cat', 0),
        (2, 'nr_class', 3),
        (2, 'nr_inst', 150),
        (2, 'nr_num', 4),
        # (0, 'num_to_cat', np.nan)
    ])
    def test_ft_methods_general(self, dt_id, ft_name, exp_value):
        """Function to test each meta-feature belongs to general group.
        """
        X, y = utils.load_xy(dt_id)
        mfe = MFE(groups=["general"],
                  features=[ft_name]).fit(X.values, y.values)
        value = mfe.extract()

        if exp_value is np.nan:
            assert value[1] is exp_value
        else:
            assert np.allclose(value[1], exp_value)
