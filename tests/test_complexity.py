"""Test module for complexity class metafeatures."""
import pytest

from pymfe.mfe import MFE
from pymfe.complexity import MFEComplexity
from tests.utils import load_xy
import numpy as np

GNAME = "complexity"


class TestComplexity:
    """TestClass dedicated to test complexity metafeatures."""

    @pytest.mark.parametrize(
        "dt_id, ft_name, exp_value, precompute",
        [
            ###################
            # Mixed data
            ###################
            (0, "c1", 1.00000000, True),
            (0, "c2", 0.00000000, True),
            # (0, "cls_coef", 0.5720338983050848, True),
            # (0, "density", 0.8359183673469388, True),
            (0, "f1", [0.98658602, 0.04443099], True),
            (0, "f1v", [0.46012273, np.nan], True),
            (0, "f2", [0.18336850, np.nan], True),
            (0, "f3", [0.76000000, np.nan], True),
            (0, "f4", [0.66000000, np.nan], True),
            # (0, "hubs", [0.5572695905805614, 0.44795209297308974], True),
            (0, "l1", [0.14747055, np.nan], True),
            (0, "l2", [0.28, np.nan], True),
            (0, "l3", [0.24, np.nan], True),
            (0, "lsc", 0.98000000, True),
            (0, "n1", 1.00000000, True),
            (0, "n2", [0.90155035, 0.12036214], True),
            (0, "n3", [1.00000000, 0], True),
            (0, "n4", [0.48, 0.50467205], True),
            (0, "t1", [0.02000000, 0], True),
            (0, "t2", 0.22000000, True),
            (0, "t3", 0.02000000, True),
            (0, "t4", 0.09090909, True),
            (0, "c1", 1.00000000, False),
            (0, "c2", 0.00000000, False),
            # (0, "cls_coef", 0.5720338983050848, False),
            # (0, "density", 0.8359183673469388, False),
            (0, "f1", [0.98658602, 0.04443099], False),
            (0, "f1v", [0.46012273, np.nan], False),
            (0, "f2", [0.18336850, np.nan], False),
            (0, "f3", [0.76000000, np.nan], False),
            (0, "f4", [0.66000000, np.nan], False),
            # (0, "hubs", [0.5572695905805614, 0.44795209297308974], False),
            (0, "l1", [0.14747055, np.nan], False),
            (0, "l2", [0.28, np.nan], False),
            (0, "l3", [0.24, np.nan], False),
            (0, "lsc", 0.98000000, False),
            (0, "n1", 1.00000000, False),
            (0, "n2", [0.90155035, 0.12036214], False),
            (0, "n3", [1.00000000, 0], False),
            (0, "n4", [0.48, 0.50467205], False),
            (0, "t1", [0.02000000, 0], False),
            (0, "t2", 0.22000000, False),
            (0, "t3", 0.02000000, False),
            (0, "t4", 0.09090909, False),
            # # # ###################
            # # # # Categorical data
            # # # ###################
            (1, "c1", 0.9985755387, True),
            (1, "c2", 0.0039403669, True),
            (1, "cls_coef", 0.4438753500119289, True),
            (1, "density", 0.8086677204095103, True),
            (1, "f1", [0.9771845579, 0.0439385223], True),
            (1, "f1v", [0.1232747983, np.nan], True),
            (1, "f2", [0.0000000000, np.nan], True),
            (1, "f3", [0.8172715895, np.nan], True),
            (1, "f4", [0.5669586984, np.nan], True),
            (1, "hubs", [0.748292972381515, 0.27590418362089725], True),
            (1, "l1", [0.026237978, np.nan], True),
            (1, "l2", [0.025969962, np.nan], True),
            (1, "l3", [0.05006258, np.nan], True),
            (1, "lsc", 0.9976595823, True),
            (1, "n1", 0.25563204005006257, True),
            (1, "n2", [0.42066583, 0.065616384], True),
            (1, "n3", [0.1558197747, 0.3627411516], True),
            (1, "n4", [0.13642053, 0.34328827], True),
            (1, "t1", [0.0035211262, 0.054066148], True),
            (1, "t2", 0.011889862327909888, True),
            (1, "t3", 0.0075093867, True),
            (1, "t4", 0.631578947368421, True),
            (1, "c1", 0.9985755387, False),
            (1, "c2", 0.0039403669, False),
            (1, "cls_coef", 0.4438753500119289, False),
            (1, "density", 0.8086677204095103, False),
            (1, "f1", [0.9771845579, 0.0439385223], False),
            (1, "f1v", [0.1232747983, np.nan], False),
            (1, "f2", [0.0000000000, np.nan], False),
            (1, "f3", [0.8172715895, np.nan], False),
            (1, "f4", [0.5669586984, np.nan], False),
            (1, "hubs", [0.748292972381515, 0.27590418362089725], False),
            (1, "l1", [0.026237978, np.nan], False),
            (1, "l2", [0.025969962, np.nan], False),
            (1, "l3", [0.05006258, np.nan], False),
            (1, "lsc", 0.9976595823, False),
            (1, "n1", 0.25563204005006257, False),
            (1, "n2", [0.42066583, 0.065616384], False),
            (1, "n3", [0.1558197747, 0.3627411516], False),
            (1, "n4", [0.13642053, 0.34328827], False),
            (1, "t1", [0.0035211262, 0.054066148], False),
            (1, "t2", 0.011889862327909888, False),
            (1, "t3", 0.0075093867, False),
            (1, "t4", 0.631578947368421, False),
            # ###################
            # # Numerical data
            # ###################
            (2, "c1", 1.000000000, True),
            (2, "c2", 0.000000000, True),
            (2, "cls_coef", 0.23776087033810933, True),
            (2, "density", 0.8170022371364654, True),
            (2, "f1", [0.279814645, 0.264900694], True),
            (2, "f1v", [0.026773189, 0.033791788], True),
            (2, "f2", [0.006381766, 0.011053544], True),
            (2, "f3", [0.123333333, 0.213619600], True),
            (2, "f4", [0.043333333, 0.075055535], True),
            (2, "hubs", [0.780120833, 0.323864856], True),
            (2, "l1", [0.004335693, 0.007509640], True),
            (2, "l2", [0.013333333, 0.023094011], True),
            (2, "l3", [0.003333333, 0.005773503], True),
            (2, "lsc", 0.816400000, True),
            (2, "n1", 0.1, True),
            (2, "n2", [0.21094362, 0.1366869], True),
            (2, "n3", [0.046666667, 0.2116305], True),
            (2, "n4", [0.013333334, 0.11508193], True),
            (2, "t1", [0.015151516, 0.024628395], True),
            (2, "t2", 0.026666667, True),
            (2, "t3", 0.013333333, True),
            (2, "t4", 0.500000000, True),
            (2, "c1", 1.000000000, False),
            (2, "c2", 0.000000000, False),
            (2, "cls_coef", 0.23776087033810933, False),
            (2, "density", 0.8170022371364654, False),
            (2, "f1", [0.279814645, 0.264900694], False),
            (2, "f1v", [0.026773189, 0.033791788], False),
            (2, "f2", [0.006381766, 0.011053544], False),
            (2, "f3", [0.123333333, 0.213619600], False),
            (2, "f4", [0.043333333, 0.075055535], False),
            (2, "hubs", [0.780120833, 0.323864856], False),
            (2, "l1", [0.004335693, 0.007509640], False),
            (2, "l2", [0.013333333, 0.023094011], False),
            (2, "l3", [0.003333333, 0.005773503], False),
            (2, "lsc", 0.816400000, False),
            (2, "n1", 0.1, False),
            (2, "n2", [0.21094362, 0.1366869], False),
            (2, "n3", [0.046666667, 0.2116305], False),
            (2, "n4", [0.013333334, 0.11508193], False),
            (2, "t1", [0.015151516, 0.024628395], False),
            (2, "t2", 0.026666667, False),
            (2, "t3", 0.013333333, False),
            (2, "t4", 0.500000000, False),
        ],
    )
    def test_ft_methods_complexity(
        self, dt_id, ft_name, exp_value, precompute
    ):
        """Function to test each meta-feature belongs to complexity group."""
        precomp_group = GNAME if precompute else None

        X, y = load_xy(dt_id)
        mfe = MFE(groups=[GNAME], features=[ft_name], random_state=1234)

        mfe.fit(X.values, y.values, precomp_groups=precomp_group)

        value = mfe.extract()[1]

        if exp_value is np.nan:
            assert value[0] is exp_value
        else:
            assert np.allclose(value, exp_value, equal_nan=True, rtol=0.025)

    @pytest.mark.parametrize(
        "num_inst_1, num_inst_2, expected_val",
        (
            (4, 0, (0, 0)),
            (0, 5, (0, 0)),
            (4, 6, (7, 6)),
        ),
    )
    def test_overlapping_area(self, num_inst_1, num_inst_2, expected_val):
        N_cls_1 = np.asarray(
            [
                [0, 0],
                [1, 1],
                [1, 0],
                [0, 1],
            ]
        )[:num_inst_1, :]

        N_cls_2 = np.asarray(
            [
                [2, 0.5],
                [0.5, 0.5],
                [0, 0],
                [-1, -1],
                [-1, 0],
                [0, -1],
            ]
        )[:num_inst_2, :]

        ind_less_overlap, feat_overlap_num, _ = MFEComplexity._calc_overlap(
            N=np.vstack((N_cls_1, N_cls_2)),
            minmax=MFEComplexity._calc_minmax(N_cls_1, N_cls_2),
            maxmin=MFEComplexity._calc_maxmin(N_cls_1, N_cls_2),
        )

        assert ind_less_overlap == np.argmin(feat_overlap_num) and np.allclose(
            feat_overlap_num, expected_val
        )

    def test_empty_minmin(self):
        arr = np.empty(shape=(0, 4))
        res = MFEComplexity._calc_minmin(arr, arr)
        assert res.size == arr.shape[1] and np.all(np.isinf(res))

    def test_empty_maxmax(self):
        arr = np.empty(shape=(0, 4))
        res = MFEComplexity._calc_maxmax(arr, arr)
        assert res.size == arr.shape[1] and np.all(np.isinf(res))

    @pytest.mark.parametrize(
        "orig_dist_mat_min, orig_dist_mat_ptp",
        [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ],
    )
    def test_t1_arguments(self, orig_dist_mat_min, orig_dist_mat_ptp):
        exp_val = [0.015151516, 0.024628395]
        X, y = load_xy(2)

        extractor = MFE(groups="complexity", features="t1")
        extractor.fit(X.values, y.values, transform_num=False)

        args = {"t1": {}}

        if not orig_dist_mat_min:
            args["t1"].update({"orig_dist_mat_min": None})

        if not orig_dist_mat_ptp:
            args["t1"].update({"orig_dist_mat_ptp": None})

        _, res = extractor.extract(**args)

        assert np.allclose(res, exp_val)
