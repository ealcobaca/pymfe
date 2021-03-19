"""Test module for statistical class metafeatures."""
import pytest

from pymfe.mfe import MFE
from tests.utils import load_xy
import numpy as np

GNAME = "statistical"


class TestStatistical:
    """TestClass dedicated to test statistical metafeatures."""

    @pytest.mark.parametrize(
        "dt_id, ft_name, exp_value, precompute",
        [
            ###################
            # Mixed data
            ###################
            (0, "can_cor", [4.967439e-01, np.nan], True),
            (0, "cor", [1.441612e-01, 1.677086e-01], True),
            (0, "cov", [7.066178e08, 5.239762e09], True),
            (0, "eigenvalues", [3.690903e12, 1.224126e13], True),
            (0, "g_mean", [148780.75, 493079.0], True),
            (0, "gravity", 1.675634e05, True),
            (0, "h_mean", [5.998783e04, 1.989364e05], True),
            (0, "iq_range", [1.920484e05, 6.339866e05], True),
            (0, "kurtosis", [7.790129e-01, 1.927274e00], True),
            (0, "lh_trace", 0.32758841958393997, True),
            (0, "mad", [1.256607e05, 4.159848e05], True),
            (0, "max", [2.069934e06, 6.837930e06], True),
            (0, "mean", [4.029463e05, 1.333427e06], True),
            (0, "median", [1.470961e05, 4.873190e05], True),
            (0, "min", [1.478355e04, 4.903048e04], True),
            (0, "nr_cor_attr", [1.818182e-02], True),
            (0, "nr_disc", 1, True),
            (0, "nr_norm", 0, True),
            (0, "nr_outliers", 11, True),
            (0, "p_trace", 0.24675450218721, True),
            (0, "range", [2.055151e06, 6.788900e06], True),
            (0, "roy_root", 0.32758839, True),
            (0, "sd", [5.807830e05, 1.920665e06], True),
            (0, "sd_ratio", np.nan, True),
            (0, "skewness", [1.563538e00, 3.244487e-01], True),
            (0, "sparsity", [9.183673e-02, 1.060439e-01], True),
            (0, "t_mean", [1.609781e05, 5.329507e05], True),
            (0, "var", [3.690903e12, 1.224125e13], True),
            (0, "w_lambda", 0.7348737, True),
            (0, "can_cor", [4.967439e-01, np.nan], False),
            (0, "cor", [1.441612e-01, 1.677086e-01], False),
            (0, "cov", [7.066178e08, 5.239762e09], False),
            (0, "eigenvalues", [3.690903e12, 1.224126e13], False),
            (0, "g_mean", [148780.75, 493079.0], False),
            (0, "gravity", 1.675634e05, False),
            (0, "h_mean", [5.998783e04, 1.989364e05], False),
            (0, "iq_range", [1.920484e05, 6.339866e05], False),
            (0, "kurtosis", [7.790129e-01, 1.927274e00], False),
            (0, "lh_trace", 0.32758841958393997, False),
            (0, "mad", [1.256607e05, 4.159848e05], False),
            (0, "max", [2.069934e06, 6.837930e06], False),
            (0, "mean", [4.029463e05, 1.333427e06], False),
            (0, "median", [1.470961e05, 4.873190e05], False),
            (0, "min", [1.478355e04, 4.903048e04], False),
            (0, "nr_cor_attr", [1.818182e-02], False),
            (0, "nr_disc", 1, False),
            (0, "nr_norm", 0, False),
            (0, "nr_outliers", 11, False),
            (0, "p_trace", 0.24675450218721, False),
            (0, "range", [2.055151e06, 6.788900e06], False),
            (0, "roy_root", 0.32758839, False),
            (0, "sd", [5.807830e05, 1.920665e06], False),
            (0, "sd_ratio", np.nan, False),
            (0, "skewness", [1.563538e00, 3.244487e-01], False),
            (0, "sparsity", [9.183673e-02, 1.060439e-01], False),
            (0, "t_mean", [1.609781e05, 5.329507e05], False),
            (0, "var", [3.690903e12, 1.224125e13], False),
            (0, "w_lambda", 0.7348737, False),
            ###################
            # Categorical data
            ###################
            (1, "can_cor", [0.79982271, np.nan], True),
            (1, "cor", [0.08564411, 0.10816678], True),
            (1, "cov", [0.01065760, 0.01849074], True),
            (1, "eigenvalues", [0.12702470, 0.15885051], True),
            (1, "g_mean", [0, 0], True),
            (1, "gravity", 0.76488534, True),
            (1, "h_mean", [0, 0], True),
            (1, "iq_range", [0.33333333, 0.47756693], True),
            (1, "kurtosis", [105.2110, 517.1173], True),
            (1, "lh_trace", 1.7755909777848424, True),
            (1, "mad", [0, 0], True),
            (1, "max", [1, 0], True),
            (1, "mean", [0.2686582, 0.2606574], True),
            (1, "median", [0.1842105, 0.3928595], True),
            (1, "min", [0, 0], True),
            (1, "nr_cor_attr", 0.01422475, True),
            (1, "nr_disc", 1, True),
            (1, "nr_norm", 0, True),
            (1, "nr_outliers", 25, True),
            (1, "p_trace", 0.6397163674317442, True),
            (1, "range", [1, 0], True),
            (1, "roy_root", 1.77559093, True),
            (1, "sd", [0.32349560, 0.15153916], True),
            (1, "sd_ratio", np.nan, True),
            (1, "skewness", [4.108820, 9.629959], True),
            (1, "sparsity", [0.49521243, 0.02778647], True),
            (1, "t_mean", [0.2248093, 0.3337982], True),
            (1, "var", [0.12702470, 0.08652912], True),
            (1, "w_lambda", 0.36028363256825574, True),
            (1, "can_cor", [0.79982271, np.nan], False),
            (1, "cor", [0.08564411, 0.10816678], False),
            (1, "cov", [0.01065760, 0.01849074], False),
            (1, "eigenvalues", [0.12702470, 0.15885051], False),
            (1, "g_mean", [0, 0], False),
            (1, "gravity", 0.76488534, False),
            (1, "h_mean", [0, 0], False),
            (1, "iq_range", [0.33333333, 0.47756693], False),
            (1, "kurtosis", [105.2110, 517.1173], False),
            (1, "lh_trace", 1.7755909777848424, False),
            (1, "mad", [0, 0], False),
            (1, "max", [1, 0], False),
            (1, "mean", [0.2686582, 0.2606574], False),
            (1, "median", [0.1842105, 0.3928595], False),
            (1, "min", [0, 0], False),
            (1, "nr_cor_attr", 0.01422475, False),
            (1, "nr_disc", 1, False),
            (1, "nr_norm", 0, False),
            (1, "nr_outliers", 25, False),
            (1, "p_trace", 0.6397163674317442, False),
            (1, "range", [1, 0], False),
            (1, "roy_root", 1.77559093, False),
            (1, "sd", [0.32349560, 0.15153916], False),
            (1, "sd_ratio", np.nan, False),
            (1, "skewness", [4.108820, 9.629959], False),
            (1, "sparsity", [0.49521243, 0.02778647], False),
            (1, "t_mean", [0.2248093, 0.3337982], False),
            (1, "var", [0.12702470, 0.08652912], False),
            (1, "w_lambda", 0.36028363256825574, False),
            ###################
            # Numerical data
            ###################
            (2, "can_cor", [0.72548576, 0.36680730], True),
            (2, "cor", [0.58981572, 0.34191469], True),
            (2, "cov", [0.59432267, 0.56030719], True),
            (2, "eigenvalues", [1.14232282, 2.05710822], True),
            (2, "g_mean", [3.22172156, 2.02456808], True),
            (2, "gravity", 3.20517457, True),
            (2, "h_mean", [2.97629003, 2.14893747], True),
            (2, "iq_range", [1.70000000, 1.27540843], True),
            (2, "kurtosis", [-0.79537400, 0.75835782], True),
            (2, "lh_trace", 32.54951329402913, True),
            (2, "mad", [1.07488500, 0.60678020], True),
            (2, "max", [5.42500000, 2.44318781], True),
            (2, "mean", [3.46366667, 1.91901800], True),
            (2, "median", [3.61250000, 1.91936404], True),
            (2, "min", [1.85000000, 1.80831413], True),
            (2, "nr_cor_attr", 0.5, True),
            (2, "nr_disc", 2, True),
            (2, "nr_norm", 1, True),
            (2, "nr_outliers", 1, True),
            (2, "p_trace", 1.1872067523722512, True),
            (2, "range", [3.57500000, 1.65000000], True),
            (2, "roy_root", 32.27195242, True),
            (2, "sd", [0.94731040, 0.57146108], True),
            (2, "sd_ratio", 1.27345134, True),
            (2, "skewness", [0.06603418, 0.29886394], True),
            (2, "sparsity", [0.02871478, 0.01103236], True),
            (2, "t_mean", [3.46972222, 1.90505400], True),
            (2, "var", [1.14232282, 1.33129110], True),
            (2, "w_lambda", 0.02352545, True),
            (2, "can_cor", [0.72548576, 0.36680730], False),
            (2, "cor", [0.58981572, 0.34191469], False),
            (2, "cov", [0.59432267, 0.56030719], False),
            (2, "eigenvalues", [1.14232282, 2.05710822], False),
            (2, "g_mean", [3.22172156, 2.02456808], False),
            (2, "gravity", 3.20517457, False),
            (2, "h_mean", [2.97629003, 2.14893747], False),
            (2, "iq_range", [1.70000000, 1.27540843], False),
            (2, "kurtosis", [-0.79537400, 0.75835782], False),
            (2, "lh_trace", 32.54951329402913, False),
            (2, "mad", [1.07488500, 0.60678020], False),
            (2, "max", [5.42500000, 2.44318781], False),
            (2, "mean", [3.46366667, 1.91901800], False),
            (2, "median", [3.61250000, 1.91936404], False),
            (2, "min", [1.85000000, 1.80831413], False),
            (2, "nr_cor_attr", 0.5, False),
            (2, "nr_disc", 2, False),
            (2, "nr_norm", 1, False),
            (2, "nr_outliers", 1, False),
            (2, "p_trace", 1.1872067523722512, False),
            (2, "range", [3.57500000, 1.65000000], False),
            (2, "roy_root", 32.27195242, False),
            (2, "sd", [0.94731040, 0.57146108], False),
            (2, "sd_ratio", 1.27345134, False),
            (2, "skewness", [0.06603418, 0.29886394], False),
            (2, "sparsity", [0.02871478, 0.01103236], False),
            (2, "t_mean", [3.46972222, 1.90505400], False),
            (2, "var", [1.14232282, 1.33129110], False),
            (2, "w_lambda", 0.02352545, False),
        ],
    )
    def test_ft_methods_statistical(
        self, dt_id, ft_name, exp_value, precompute
    ):
        """Function to test each meta-feature belongs to statistical group."""
        precomp_group = GNAME if precompute else None
        X, y = load_xy(dt_id)
        mfe = MFE(groups=[GNAME], features=[ft_name]).fit(
            X.values, y.values, precomp_groups=precomp_group
        )
        value = mfe.extract()[1]

        assert np.allclose(
            value, exp_value, atol=0.001, rtol=0.05, equal_nan=True
        )

    @pytest.mark.parametrize(
        "dt_id, exp_value, precompute, test, failure",
        [
            (0, 0, False, "shapiro-wilk", "soft"),
            (1, 0, False, "shapiro-wilk", "soft"),
            (2, 1, False, "shapiro-wilk", "soft"),
            (0, 0, True, "shapiro-wilk", "soft"),
            (1, 0, True, "shapiro-wilk", "soft"),
            (2, 1, True, "shapiro-wilk", "soft"),
            (0, 0, False, "dagostino-pearson", "soft"),
            (1, 0, False, "dagostino-pearson", "soft"),
            (2, 2, False, "dagostino-pearson", "soft"),
            (0, 0, True, "dagostino-pearson", "soft"),
            (1, 0, True, "dagostino-pearson", "soft"),
            (2, 2, True, "dagostino-pearson", "soft"),
            (0, 0, False, "anderson-darling", "soft"),
            (1, 0, False, "anderson-darling", "soft"),
            (2, 2, False, "anderson-darling", "soft"),
            (0, 0, True, "anderson-darling", "soft"),
            (1, 0, True, "anderson-darling", "soft"),
            (2, 2, True, "anderson-darling", "soft"),
            (0, 0, False, "all", "soft"),
            (1, 0, False, "all", "soft"),
            (2, 2, False, "all", "soft"),
            (0, 0, True, "all", "soft"),
            (1, 0, True, "all", "soft"),
            (2, 2, True, "all", "soft"),
            (0, 0, False, "all", "hard"),
            (1, 0, False, "all", "hard"),
            (2, 1, False, "all", "hard"),
            (0, 0, True, "all", "hard"),
            (1, 0, True, "all", "hard"),
            (2, 1, True, "all", "hard"),
        ],
    )
    def test_normality_tests(
        self, dt_id, exp_value, precompute, test, failure
    ):
        """Test normality tests included in ``nr_norm`` statistical method."""
        precomp_group = GNAME if precompute else None
        X, y = load_xy(dt_id)
        mfe = MFE(groups=[GNAME], features="nr_norm").fit(
            X.values, y.values, precomp_groups=precomp_group
        )
        value = mfe.extract(nr_norm={"failure": failure, "method": test})[1]

        assert np.allclose(
            value, exp_value, atol=0.001, rtol=0.05, equal_nan=True
        )

    @pytest.mark.parametrize(
        "test, failure",
        [
            ("invalid", "soft"),
            ("anderson-darling", "invalid"),
            ("invalid", "invalid"),
            (None, "soft"),
            ("all", None),
        ],
    )
    def test_error_normality_tests(self, test, failure):
        with pytest.warns(RuntimeWarning):
            X, y = load_xy(0)
            mfe = MFE(groups=[GNAME], features="nr_norm")
            mfe.fit(X.values, y.values, precomp_groups=None)
            mfe.extract(nr_norm={"failure": failure, "method": test})

    def test_none_cancor(self):
        X, y = load_xy(0)

        feats = [
            "w_lambda",
            "p_trace",
            "lh_trace",
            "roy_root",
        ]

        mfe = MFE(groups=[GNAME], features=feats)

        custom_args = {
            "can_cors": np.array([]),
            "can_cor_eigvals": np.array([]),
        }

        mfe.fit(X.values, y.values, precomp_groups=None)

        extract_args = {cur_feat: custom_args for cur_feat in feats}
        vals = mfe.extract(**extract_args, suppress_warnings=True)[1]

        assert np.allclose(
            vals, np.full(shape=len(vals), fill_value=np.nan), equal_nan=True
        )

    @pytest.mark.parametrize(
        "dt_id, exp_value, precompute, criterion",
        [
            (0, 0.32758839, False, "eigval"),
            (0, 0.24675448231745442, False, "cancor"),
            (1, 1.77559093, False, "eigval"),
            (1, 0.6397163674317442, False, "cancor"),
            (2, 32.27195242, False, "eigval"),
            (2, 0.9699446498549823, False, "cancor"),
            (0, 0.32758839, True, "eigval"),
            (0, 0.24675448231745442, True, "cancor"),
            (1, 1.77559093, True, "eigval"),
            (1, 0.6397163674317442, True, "cancor"),
            (2, 32.27195242, True, "eigval"),
            (2, 0.9699446498549823, True, "cancor"),
        ],
    )
    def test_roy_largest_root(self, dt_id, exp_value, precompute, criterion):
        precomp_group = GNAME if precompute else None
        X, y = load_xy(dt_id)
        mfe = MFE(groups=[GNAME], features="roy_root").fit(
            X.values, y.values, precomp_groups=precomp_group
        )
        value = mfe.extract(roy_root={"criterion": criterion})[1]

        assert np.allclose(
            value, exp_value, atol=0.001, rtol=0.05, equal_nan=True
        )

    @pytest.mark.parametrize("criterion", ("invalid", "", None))
    def test_roy_largest_root_invalid_criteria(self, criterion):
        with pytest.warns(RuntimeWarning):
            X, y = load_xy(0)
            mfe = MFE(groups=[GNAME], features="roy_root")
            mfe.fit(X.values, y.values, precomp_groups=None)
            mfe.extract(roy_root={"criterion": criterion})

    @pytest.mark.parametrize(
        "dt_id, exp_value, precompute",
        [
            (
                0,
                [
                    4.967439e-01,
                    1.441612e-01,
                    7.066178e08,
                    3.690903e12,
                    148780.75,
                    1.675634e05,
                    5.998783e04,
                    1.920484e05,
                    7.790129e-01,
                    0.32758841958393997,
                    1.256607e05,
                    2.069934e06,
                    4.029463e05,
                    1.470961e05,
                    1.478355e04,
                    1.818182e-02,
                    1,
                    0,
                    11,
                    0.24675450218721,
                    2.055151e06,
                    0.32758839,
                    5.807830e05,
                    np.nan,
                    1.563538e00,
                    9.183673e-02,
                    1.609781e05,
                    3.690903e12,
                    0.7348737,
                ],
                False,
            ),
            (
                0,
                [
                    4.967439e-01,
                    1.441612e-01,
                    7.066178e08,
                    3.690903e12,
                    148780.75,
                    1.675634e05,
                    5.998783e04,
                    1.920484e05,
                    7.790129e-01,
                    0.32758841958393997,
                    1.256607e05,
                    2.069934e06,
                    4.029463e05,
                    1.470961e05,
                    1.478355e04,
                    1.818182e-02,
                    1,
                    0,
                    11,
                    0.24675450218721,
                    2.055151e06,
                    0.32758839,
                    5.807830e05,
                    np.nan,
                    1.563538e00,
                    9.183673e-02,
                    1.609781e05,
                    3.690903e12,
                    0.7348737,
                ],
                True,
            ),
            (
                2,
                [
                    0.72548576,
                    0.58981572,
                    0.59432267,
                    1.14232282,
                    3.22172156,
                    3.20517457,
                    2.97629003,
                    1.70000000,
                    -0.79537400,
                    32.54951329402913,
                    1.07488500,
                    5.42500000,
                    3.46366667,
                    3.61250000,
                    1.85000000,
                    0.5,
                    2,
                    1,
                    1,
                    1.1872067523722512,
                    3.57500000,
                    32.27195242,
                    0.94731040,
                    1.27345134,
                    0.06603418,
                    0.02871478,
                    3.46972222,
                    1.14232282,
                    0.02352545,
                ],
                False,
            ),
            (
                2,
                [
                    0.72548576,
                    0.58981572,
                    0.59432267,
                    1.14232282,
                    3.22172156,
                    3.20517457,
                    2.97629003,
                    1.70000000,
                    -0.79537400,
                    32.54951329402913,
                    1.07488500,
                    5.42500000,
                    3.46366667,
                    3.61250000,
                    1.85000000,
                    0.5,
                    2,
                    1,
                    1,
                    1.1872067523722512,
                    3.57500000,
                    32.27195242,
                    0.94731040,
                    1.27345134,
                    0.06603418,
                    0.02871478,
                    3.46972222,
                    1.14232282,
                    0.02352545,
                ],
                True,
            ),
        ],
    )
    def test_integration_statistical(self, dt_id, exp_value, precompute):
        """Function to test all statistical meta-features simultaneously."""
        precomp_group = GNAME if precompute else None
        X, y = load_xy(dt_id)
        mfe = MFE(groups=[GNAME], summary="mean").fit(
            X.values, y.values, precomp_groups=precomp_group
        )
        value = mfe.extract()[1]

        assert np.allclose(
            value, exp_value, atol=0.001, rtol=0.05, equal_nan=True
        )
