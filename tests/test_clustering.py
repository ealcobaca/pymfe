"""Test module for clustering class metafeatures."""
import pytest

from pymfe.mfe import MFE
from pymfe.clustering import MFEClustering
from tests.utils import load_xy
import numpy as np

GNAME = "clustering"


class TestClustering:
    """TestClass dedicated to test clustering based metafeatures."""

    @pytest.mark.parametrize(
        "dt_id, ft_name, exp_value, precompute",
        [
            ###################
            # Mixed data
            ###################
            (0, "ch", 0.008469636865711082, False),
            (0, "int", 5728840.510362266, False),
            (0, "nre", 0.6931471805599453, False),
            (0, "pb", 0.016754815003958073, False),
            (0, "sc", 0, False),
            (0, "sil", -0.03842692011975991, False),
            (0, "vdb", 58.22425419399301, False),
            (0, "vdu", 1.698593922818614e-08, False),
            (0, "ch", 0.008469636865711082, True),
            (0, "int", 5728840.510362266, True),
            (0, "nre", 0.6931471805599453, True),
            (0, "pb", 0.016754815003958073, True),
            (0, "sc", 0, True),
            (0, "sil", -0.03842692011975991, True),
            (0, "vdb", 58.22425419399301, True),
            (0, "vdu", 1.698593922818614e-08, True),
            ###################
            # Categorical data
            ###################
            (1, "ch", 97.00317235631576, False),
            (1, "int", 3.153969124655814, False),
            (1, "nre", 0.6921598191951595, False),
            (1, "pb", -0.09651355252305963, False),
            (1, "sc", 0, False),
            (1, "sil", 0.0316248093980556, False),
            (1, "vdb", 5.676987002418947, False),
            (1, "vdu", 8.009381946424901e-08, False),
            (1, "ch", 97.00317235631576, True),
            (1, "int", 3.153969124655814, True),
            (1, "nre", 0.6921598191951595, True),
            (1, "pb", -0.09651355252305963, True),
            (1, "sc", 0, True),
            (1, "sil", 0.0316248093980556, True),
            (1, "vdb", 5.676987002418947, True),
            (1, "vdu", 8.009381946424901e-08, True),
            ###################
            # Numerical data
            ###################
            (2, "ch", 486.32083931855703, False),
            (2, "int", 3.321079768101941, False),
            (2, "nre", 1.0986122886681096, False),
            (2, "pb", -0.6798579850365509, False),
            (2, "sc", 0, False),
            (2, "sil", 0.5032506980366624, False),
            (2, "vdb", 0.7517428073901388, False),
            (2, "vdu", 2.3392212797698888e-05, False),
            (2, "ch", 486.32083931855703, True),
            (2, "int", 3.321079768101941, True),
            (2, "nre", 1.0986122886681096, True),
            (2, "pb", -0.6798579850365509, True),
            (2, "sc", 0, True),
            (2, "sil", 0.5032506980366624, True),
            (2, "vdb", 0.7517428073901388, True),
            (2, "vdu", 2.3392212797698888e-05, True),
        ],
    )
    def test_ft_methods_clustering(
        self, dt_id, ft_name, exp_value, precompute
    ):
        """Function to test each meta-feature belongs to clustering group."""

        precomp_group = GNAME if precompute else None
        X, y = load_xy(dt_id)
        mfe = MFE(groups=[GNAME], features=[ft_name]).fit(
            X.values, y.values, precomp_groups=precomp_group
        )
        value = mfe.extract()[1]

        if exp_value is np.nan:
            assert value[0] is exp_value

        else:
            assert np.allclose(value, exp_value)

    @pytest.mark.parametrize("precompute", [False, True])
    def test_silhouette_subsampling(self, precompute):
        X, y = load_xy(0)
        precomp_group = GNAME if precompute else None
        mfe = MFE(groups="clustering", features="sil", random_state=1234).fit(
            X.values, y.values, precomp_groups=precomp_group
        )
        value = mfe.extract(sil={"sample_frac": 0.5})[1]

        assert np.allclose(value, -0.07137712254830314)

    @staticmethod
    def test_precompute_nearest_neighbors():
        N = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])
        aux = MFEClustering.precompute_nearest_neighbors(
            N, y, class_freqs=None
        )

        assert isinstance(aux, dict)
        assert len(aux) == 1
        assert np.allclose(aux["nearest_neighbors"], np.array([[1], [0]]))

    @staticmethod
    def test_errors_get_class_representatives():
        N = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])
        with pytest.raises(ValueError):
            MFEClustering._get_class_representatives(N, y, representative="42")

        with pytest.raises(TypeError):
            MFEClustering._get_class_representatives(N, y, representative=1)

    @pytest.mark.parametrize(
        "dt_id, exp_value, precompute",
        [
            ###################
            # Mixed data
            ###################
            (
                0,
                [
                    0.008469636865711082,
                    5728840.510362266,
                    0.6931471805599453,
                    0.016754815003958073,
                    0,
                    -0.03842692011975991,
                    58.22425419399301,
                    1.698593922818614e-08,
                ],
                False,
            ),
            (
                0,
                [
                    0.008469636865711082,
                    5728840.510362266,
                    0.6931471805599453,
                    0.016754815003958073,
                    0,
                    -0.03842692011975991,
                    58.22425419399301,
                    1.698593922818614e-08,
                ],
                True,
            ),
            ###################
            # Numerical data
            ###################
            (
                2,
                [
                    486.32083931855703,
                    3.321079768101941,
                    1.0986122886681096,
                    -0.6798579850365509,
                    0,
                    0.5032506980366624,
                    0.7517428073901388,
                    2.3392212797698888e-05,
                ],
                False,
            ),
            (
                2,
                [
                    486.32083931855703,
                    3.321079768101941,
                    1.0986122886681096,
                    -0.6798579850365509,
                    0,
                    0.5032506980366624,
                    0.7517428073901388,
                    2.3392212797698888e-05,
                ],
                True,
            ),
        ],
    )
    def test_integration_clustering(self, dt_id, exp_value, precompute):
        """Function to test each all clustering meta-features."""

        precomp_group = GNAME if precompute else None
        X, y = load_xy(dt_id)
        mfe = MFE(groups=[GNAME], summary="mean").fit(
            X.values, y.values, precomp_groups=precomp_group
        )
        value = mfe.extract()[1]

        assert np.allclose(value, exp_value, equal_nan=True)
