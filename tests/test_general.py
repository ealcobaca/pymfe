"""Test module for General class metafeatures."""
import pytest

import arff
import pandas as pd
from pymfe.mfe import MFE

DATA_ID = [
    "tests/test_datasets/mix_aids.arff"
]

class TestGeneral:
    """TestClass dedicated to test General metafeatures."""
    @pytest.mark.parametrize("dt_id, g_name, ft_name, exp_value", [
        (0, 'general', 'freq_class', [0.5, 0.0])
    ])
    def test_ft_methods_general(self, dt_id, g_name, ft_name,
                                exp_value):
        data = arff.load(open(DATA_ID[dt_id], 'r'))
        df = pd.DataFrame(data['data'])
        y = df.iloc[:, -1]
        X = df.iloc[:, :df.shape[1]-1]

        mfe = MFE(groups=[g_name], features=[ft_name])
        mfe.fit(X.values, y.values)
        value = mfe.extract()

        assert value[1] == exp_value
