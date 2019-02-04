"""Test module for core class MFE and _internal module."""
import pytest
import collections

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
        """Tests 'group' param, valid iterable input."""
        mfe_groups = set(_internal.process_generic_set(groups, "groups"))
        assert not mfe_groups.difference(expected)

    @pytest.mark.parametrize("groups, expected", (
        ("info-theory", ("info-theory", )),
        ("statistical", ("statistical", )),
        ("general", ("general", )),
        ("model-based", ("model-based", )),
        ("landmarking", ("landmarking", )),
    ))
    def test_param_groups_single_valid(self, groups, expected):
        """Tests 'group' param, valid single-valued input."""
        mfe_groups = set(_internal.process_generic_set(groups, "groups"))
        assert not mfe_groups.difference(expected)

    @pytest.mark.parametrize("timeopt, expected", (
        (None, None),
        ("AVG", "avg"),
        ("avg_summ", "avg_summ"),
        ("Total_Summ", "total_summ"),
        ("TOTAL", "total"),
    ))
    def test_param_timeopt_single_valid(self, timeopt, expected):
        """Tests 'group' param, valid single-valued input."""
        mfe_timeopt = _internal.process_generic_option(
            timeopt, "timeopt", allow_none=True)

        assert mfe_timeopt == expected

    @pytest.mark.parametrize("rescale, expected", (
        (None, None),
        ("ROBUST", "robust"),
        ("Min-Max", "min-max"),
        ("Standard", "standard"),
    ))
    def test_param_rescale_single_valid(self, rescale, expected):
        """Tests 'group' param, valid single-valued input."""
        mfe_rescale = _internal.process_generic_option(
            rescale, "rescale", allow_none=True)

        assert mfe_rescale == expected

    @pytest.mark.parametrize("timeopt, expected_exception", (
        (("total", ), TypeError),
        ([None], TypeError),
        ("totaal", ValueError),
        ("", ValueError),
        ("all", ValueError),
        ("avg_", ValueError),
    ))
    def test_param_timeopt_notvalid(self, timeopt, expected_exception):
        """Tests 'timeopt' param, invalid input."""
        with pytest.raises(expected_exception):
            _internal.process_generic_option(
                timeopt, "timeopt", allow_none=True)

    @pytest.mark.parametrize("rescale, expected_exception", (
        (("standard", ), TypeError),
        ([None], TypeError),
        ("robustrobust", ValueError),
        ("all", ValueError),
        ("", ValueError),
        ("min_max", ValueError),
    ))
    def test_param_rescale_notvalid(self, rescale, expected_exception):
        """Tests 'rescale' param, invalid input."""
        with pytest.raises(expected_exception):
            _internal.process_generic_option(
                rescale, "rescale", allow_none=True)

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
        """Tests 'group' param, invalid iterable input."""
        with pytest.raises(expected_exception):
            _internal.process_generic_set(groups, "groups")

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
        """Tests 'group' param, invalid single-val input."""
        with pytest.raises(expected_exception):
            _internal.process_generic_set(groups, "groups")


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
                "ddof": 2,
            }, 1.2423693),
            (DATA_GENERIC_NUMERIC_1, _summary.SUMMARY_METHODS["sd"], {
                "ddof": 1,
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
                "bins": 3,
                "normalize": False,
            }, [4, 3, 1]),
            (DATA_GENERIC_NUMERIC_1, _summary.SUMMARY_METHODS["histogram"], {
                "bins": 5,
                "normalize": True,
            }, [0.25, 0.125, 0.25, 0.25, 0.125]),
            (DATA_GENERIC_NUMERIC_0, _summary.SUMMARY_METHODS["quantiles"],
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

    @pytest.mark.parametrize("timeopt",
                             ("total", "avg", "total_sum", "avg_summ"))
    def test_check_timeopt_working(self, timeopt):
        features = ("mean", "kurtosis", "attr_ent")
        summary = ("range", "skewness", "mean", "max")
        res = MFE(
            features=features, summary=summary, measure_time="total").fit(
                X=[1], y=[1]).extract(suppress_warnings=True)

        name, val, time = res

        assert (len(res) == 3 and len(name) == len(val) == len(time)
                and len(time) == len(features) * len(summary))

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

    @pytest.mark.parametrize(
        "dt_id, class_ind, ind_num, ind_cat, cat_cols, check_bool", (
            (1, 6, (0, 3), (1, 2, 4, 5), "auto", True),
            (1, 6, (0, 3, 4), (1, 2, 5), "auto", False),
            (0, 20, tuple(range(20)), tuple(), [], False),
            (0, 20, tuple(range(20)), tuple(), None, False),
        ))
    def test_check_num_cat_cols_indexes(self, dt_id, class_ind, ind_num,
                                        ind_cat, cat_cols, check_bool):
        """Test column indexes separated by numeric and categorical types."""
        dataset = context.DATASET_LIST[dt_id]

        if not isinstance(class_ind, collections.Iterable):
            class_ind = [class_ind]

        attr_ind = list(set(range(dataset.shape[1])).difference(class_ind))

        X = dataset.iloc[:, attr_ind].values
        y = dataset.iloc[:, class_ind].values

        model = MFE().fit(X=X, y=y, cat_cols=cat_cols, check_bool=check_bool)

        assert (sorted(model._attr_indexes_num) == sorted(ind_num)
                and sorted(model._attr_indexes_cat) == sorted(ind_cat))


class TestInternalFunctions:
    """Test avulse _internal module functions."""

    @pytest.mark.parametrize("value, check_subtype", (
        (1, False),
        (1, True),
        (3.1415, False),
        (3.1415, True),
        (-0, False),
        (-0.5, False),
        ([-0.5], True),
        ([1, 2.1, 0.0, -0.0, -.1, +1.2, +.9, 0.14, 3.1415], True),
    ))
    def test_isnumeric_valid(self, value, check_subtype):
        assert _internal.isnumeric(value=value, check_subtype=check_subtype)

    @pytest.mark.parametrize("value, check_subtype", (
        ("1", False),
        (["1"], True),
        ([1], False),
        (None, False),
        (None, True),
        ([], True),
        ([], False),
        ([], False),
        ([], True),
        ("1.2", False),
        ([3.1415], False),
        ("a", True),
        ([None, 1, 2], True),
        ([None, 1, 2], True),
        (["-.32", "1.0", "+2.0"], True),
        ([1, 2.1, 0.0, -0.0, -.1, +1.2, +.9, 0.14, 3.1415, "b"], True),
        ([1, 2.1, 0.0, -0.0, -.1, +1.2, "+.9", 0.14, "3.1415"], True),
    ))
    def test_isnumeric_invalid(self, value, check_subtype):
        assert not _internal.isnumeric(
            value=value, check_subtype=check_subtype)
