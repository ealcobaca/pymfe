"""Test module for core class MFE and _internal module."""
import pytest

import context  # noqa: F401
from pymfe.mfe import MFE
from pymfe import _internal
from pymfe import _summary


class TestMFEGroups:
    """Dedicated to test `Group` related methods/functions from MFE."""

    @pytest.mark.parametrize("groups, expected", (
        (["all"], _internal.VALID_GROUPS),
        (["ALL"], _internal.VALID_GROUPS),
        (("landmarking", "statistical", "all"), _internal.VALID_GROUPS),
        ({"landmarking", "info-theory"}, ("landmarking", "info-theory")),
        (("statistical", "general"), ("general", "statistical")),
        (("landMARKING", ), ("landmarking", )),
        (("MODEL-based", "Statistical", "GENERAL", "info-THEORY",
          "LaNdMaRKiNg"), _internal.VALID_GROUPS),
    ))
    def test_param_groups_iterable_valid(self, groups, expected):
        """Tests 'group' param (_process_groups), valid iterable input."""
        mfe_groups = set(_internal.process_groups(groups))
        assert not mfe_groups.difference(expected)

    @pytest.mark.parametrize("groups, expected", (
        ("info-theory", ("info-theory", )),
        ("statistical", ("statistical", )),
        ("general", ("general", )),
        ("model-based", ("model-based", )),
        ("landmarking", ("landmarking", )),
    ))
    def test_param_groups_single_valid(self, groups, expected):
        """Tests 'group' param (_process_groups), valid single-valued input."""
        mfe_groups = set(_internal.process_groups(groups))
        assert not mfe_groups.difference(expected)

    @pytest.mark.parametrize("groups, expected_exception", (
        (["unknown"], ValueError),
        (["allall"], ValueError),
        (["statistical", "basic"], ValueError),
        (["MODEL-BASED", "info_theory"], ValueError),
        (("landMARKING", "statisticall"), ValueError),
        ([""], ValueError),
        ([], ValueError),
        (12, TypeError),
    ))
    def test_param_groups_iterable_notvalid(self, groups, expected_exception):
        """Tests 'group' param (_process_groups), invalid iterable input."""
        with pytest.raises(expected_exception):
            set(_internal.process_groups(groups))

    @pytest.mark.parametrize("groups, expected_exception", (
        ("unknown", ValueError),
        ("allall", ValueError),
        ("", ValueError),
        (None, ValueError),
        (" ", ValueError),
        ("info_theory", ValueError),
        ("model based", ValueError),
        (123, TypeError),
    ))
    def test_param_groups_single_notvalid(self, groups, expected_exception):
        """Tests 'group' param (_process_groups), invalid single-val input."""
        with pytest.raises(expected_exception):
            set(_internal.process_groups(groups))


class TestMFESummary:
    """Dedicated to test `Summary` related methods/functions from MFE."""

    DATA_GENERIC_NUMERIC_0 = [1, 1, 2, 2, 1, 3, 1, 2]
    DATA_GENERIC_NUMERIC_1 = [
        1.01948868,
        0.09289884,
        0.26322793,
        -1.380858,
        1.90624969,
        0.80389912,
        -1.36604076,
        -0.43830465,
    ]
    EPSILON = 1.0E-6

    @pytest.mark.parametrize("summary, expected", (
        ("mean", ("mean", )),
        ("sd", ("sd", )),
        ("all", _internal.VALID_SUMMARY),
        (None, tuple()),
        ("", tuple()),
        ([], tuple()),
    ))
    def test_param_summary_single_valid(self, summary, expected):
        """Tests 'summary' (_process_summary), valid single-valued input."""
        summary_mtd_names, _ = _internal.process_summary(summary)
        mfe_summary = set(summary_mtd_names)
        assert not mfe_summary.difference(expected)

    @pytest.mark.parametrize("summary, expected", (
        (("mean", "sd"), (
            "mean",
            "sd",
        )),
        (("MeAn", "SD"), (
            "mean",
            "sd",
        )),
        (("HISTOGRAM", ), ("histogram", )),
        (("all", ), _internal.VALID_SUMMARY),
        (("mean", "ALL"), _internal.VALID_SUMMARY),
    ))
    def test_param_summary_iterable_valid(self, summary, expected):
        """Tests 'summary' (_process_summary), valid iterable input."""
        summary_mtd_names, _ = _internal.process_summary(summary)
        mfe_summary = set(summary_mtd_names)
        assert not mfe_summary.difference(expected)

    @pytest.mark.parametrize("summary, expected_exception", (
        ("unknown", ValueError),
        (" ", ValueError),
        ("meann", ValueError),
        ("allall", ValueError),
        (123, TypeError),
    ))
    def test_param_summary_single_notvalid(self, summary, expected_exception):
        """Tests 'summary' (_process_summary), invalid single-val input."""
        with pytest.raises(expected_exception):
            set(_internal.process_summary(summary))

    @pytest.mark.parametrize("summary, expected_exception", (
        (("mean", "sd", "unknown"), ValueError),
        (("sd", ""), ValueError),
        (("mean", None), TypeError),
        (("mean", "sd", 123), TypeError),
        (("mean", "sd", 123), TypeError),
        (("all", 123), TypeError),
    ))
    def test_param_summary_iterable_notvalid(self, summary,
                                             expected_exception):
        """Tests 'summary' (_process_summary), invalid iterable input."""
        with pytest.raises(expected_exception):
            set(_internal.process_summary(summary))

    @pytest.mark.parametrize(
        "features, callable_sum, callable_args, expected_value", (
            (DATA_GENERIC_NUMERIC_0, _summary.SUMMARY_METHODS["mean"], None,
             1.625),
            (DATA_GENERIC_NUMERIC_1, _summary.SUMMARY_METHODS["sd"], {
                "ddof": 2
            }, 1.2423693),
            (DATA_GENERIC_NUMERIC_1, _summary.SUMMARY_METHODS["sd"], {
                "ddof": 1
            }, 1.1502104),
        ))
    def test_summarize_single_feat_value(self, features, callable_sum,
                                         callable_args, expected_value):
        """Summarize function with callables that return a single value."""

        sum_val = _internal.summarize(
            features=features,
            callable_sum=callable_sum,
            callable_args=callable_args)

        assert abs(sum_val - expected_value) < TestMFESummary.EPSILON

    @pytest.mark.parametrize(
        "features, callable_sum, callable_args, expected_value", (
            (DATA_GENERIC_NUMERIC_0, _summary.SUMMARY_METHODS["histogram"], {
                "bins": 3
            }, [4, 3, 1]),
            (DATA_GENERIC_NUMERIC_1, _summary.SUMMARY_METHODS["histogram"], {
                "bins": 5,
                "normalize": True
            }, [0.38027352, 0.19013676, 0.38027352, 0.38027352, 0.19013676]),
            (DATA_GENERIC_NUMERIC_0, _summary.SUMMARY_METHODS["quartiles"],
             None, [1.0, 1.0, 1.5, 2.0, 3.0]),
        ))
    def test_summarize_multi_feat_value(self, features, callable_sum,
                                        callable_args, expected_value):
        """Summarize function with callables that return a multiple values."""

        sum_val = _internal.summarize(
            features=features,
            callable_sum=callable_sum,
            callable_args=callable_args)

        assert all([
            abs(a - b) < TestMFESummary.EPSILON
            for a, b in zip(sum_val, expected_value)
        ])


class TestMFEInstantiation:
    """Test cases after MFE Instantiation."""

    @pytest.mark.parametrize("features, summary, groups, expected_error", (
        ("all", "invalid_sum", "all", ValueError),
        ("all", "all", "invalid_group", ValueError),
        (None, "all", "all", ValueError),
        ("all", "all", None, ValueError),
    ))
    def test_instantiation_errors(self, features, summary, groups,
                                  expected_error):
        """Tests MFE Erros at instantiation time."""

        with pytest.raises(expected_error):
            MFE(summary=summary, groups=groups, features=features)

    @pytest.mark.parametrize("features, summary, groups, expected_warning", (
        ("invalid_feat", "all", "all", UserWarning),
        (("nr_inst", "invalid_feat"), "all", "all", UserWarning),
    ))
    def test_instantiation_warnings(self, features, summary, groups,
                                    expected_warning):
        """Test MFE warnings at instantiation time."""

        with pytest.warns(expected_warning):
            MFE(summary=summary, groups=groups, features=features)

    @pytest.mark.parametrize("features, summary, groups, suppress_warnings, "
                             "kwargs, expected_warning", (
                                 ("mean", "all", "statistical", False, {
                                     "mean": {
                                         "unk": 2,
                                     },
                                 }, UserWarning),
                                 ("all", "sd", "all", False, {
                                     "sd": {
                                         "ddof": 1,
                                         "inv": -1,
                                     },
                                 }, UserWarning),
                                 ("mean", "all", "statistical", True, {
                                     "mean": {
                                         "unk": 2,
                                     },
                                 }, None),
                             ))
    def test_extract_warnings(self, features, summary, groups,
                              suppress_warnings, kwargs, expected_warning):
        """Test extract(...) method warnings."""

        if kwargs is None:
            kwargs = {}

        with pytest.warns(expected_warning):
            model = MFE(summary=summary, groups=groups, features=features)
            model.fit(X=[1], y=[1])
            model.extract(suppress_warnings=suppress_warnings, **kwargs)

    @pytest.mark.parametrize("X, y, splits, expected_error", (
        ([1, 2, 3, 4], [1, 2, 3], None, ValueError),
        ([[1, 2, 3, 4]], [1, 2, 3, 4], None, ValueError),
        ([1, 2, 4], [0, 1, 2, 3], None, ValueError),
        ([1], [0], "a", TypeError),
        ([1], [0], 1, TypeError),
        ([1, 2, 4], None, None, TypeError),
        ([1, 2, 4], [], None, ValueError),
        (None, [0, 1, 2, 3], None, TypeError),
        ([], [0, 1, 2, 3], None, ValueError),
        ([], [], None, ValueError),
        ([[1, 2, 3], [1, 2, 4]], [0, 1, 2], None, ValueError),
    ))
    def test_fit_errors(self, X, y, splits, expected_error):
        """Test incorrect cases for fit(...) method."""

        with pytest.raises(expected_error):
            MFE().fit(X=X, y=y, splits=splits)

    @pytest.mark.parametrize("X, y, splits, shape_y, shape_X", (
        ([1, 1, 1], [0, 0, 0], None, (3, ), (3, 1)),
        ([[1, 1, 1], [1, 2, 3]], [0, 0], None, (2, ), (2, 3)),
    ))
    def test_fit_correct(self, X, y, splits, shape_y, shape_X):
        """Test correct cases for fit(...) method."""

        model = MFE().fit(X=X, y=y, splits=splits)

        assert (model.X.shape == shape_X and model.y.shape == shape_y)
