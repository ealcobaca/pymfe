"""Test module for MFE class errors and warnings."""
import pytest
import numpy as np

from pymfe.mfe import MFE
from pymfe import _internal
from pymfe import _bootstrap
from tests.utils import load_xy

GNAME = "errors-warnings"


class TestErrorsWarnings:
    """TestClass dedicated to test General metafeatures."""

    def test_error_empty_data_1(self):
        with pytest.raises(TypeError):
            MFE().fit(X=None, y=None)

    def test_error_sample_size(self):
        with pytest.raises(ValueError):
            MFE(lm_sample_frac=-1)

    def test_error_empty_data_2(self):
        with pytest.raises(TypeError):
            X, y = load_xy(0)
            model = MFE().fit(X=X.values, y=y.values)
            model.X = None
            model.extract()

    def test_error_empty_data_3(self):
        with pytest.raises(ValueError):
            MFE().fit(X=[], y=[])

    def test_error_data_wrong_shape(self):
        with pytest.raises(ValueError):
            X, y = load_xy(0)
            MFE().fit(X=X.values, y=y.values[:-1])

    @pytest.mark.parametrize(
        "group_name",
        [
            "land-marking",
            "infotheo",
            "generalgeneral",
            "generalstatistical",
            ("general", "statistical", "invalid"),
            ("invalid",),
            0,
            None,
            [],
            set(),
            tuple(),
        ],
    )
    def test_error_invalid_groups_1(self, group_name):
        with pytest.raises(ValueError):
            MFE(groups=group_name)

    @pytest.mark.parametrize(
        "group_name",
        [
            1,
            lambda x: x,
            range(1, 5),
        ],
    )
    def test_error_invalid_groups_2(self, group_name):
        with pytest.raises(TypeError):
            MFE(groups=group_name)

    def test_error_random_state(self):
        with pytest.raises(ValueError):
            MFE(random_state=1.5)

    def test_error_folds(self):
        with pytest.raises(ValueError):
            MFE(num_cv_folds=1.5)

    def test_error_cat_cols_1(self):
        with pytest.raises(ValueError):
            X, y = load_xy(0)
            MFE().fit(X=X.values, y=y.values, cat_cols=1)

    def test_error_cat_cols_2(self):
        with pytest.raises(ValueError):
            X, y = load_xy(0)
            MFE().fit(X=X.values, y=y.values, cat_cols="all")

    def test_error_invalid_timeopt(self):
        with pytest.raises(ValueError):
            X, y = load_xy(0)
            MFE(measure_time="invalid").fit(X=X.values, y=y.values)

    @pytest.mark.parametrize(
        "value, group_name, allow_none, allow_empty",
        [
            (None, "groups", False, True),
            (None, "groups", False, False),
            ("", "group", False, False),
            ("", "group", True, False),
            ("invalid", "groups", False, False),
            ("all", "invalid", False, False),
            ("invalid", "groups", False, True),
            ("invalid", "groups", True, False),
            ("invalid", "groups", True, True),
            ("mean", "summary", True, True),
            ("all", "summary", True, True),
            ("num_inst", "features", True, True),
            ("all", "features", True, True),
        ],
    )
    def test_error_process_generic_option_1(
        self, value, group_name, allow_none, allow_empty
    ):
        with pytest.raises(ValueError):
            _internal.process_generic_option(
                value=value,
                group_name=group_name,
                allow_none=allow_none,
                allow_empty=allow_empty,
            )

    def test_error_process_generic_option_2(self):
        with pytest.raises(TypeError):
            _internal.process_generic_option(values=[1, 2, 3], group_name=None)

    def test_error_process_generic_option_3(self):
        with pytest.raises(TypeError):
            _internal.process_generic_option(
                values=[1, 2, 3], group_name="timeopt"
            )

    @pytest.mark.parametrize(
        "values, group_name, allow_none, allow_empty",
        [
            (None, "groups", False, True),
            (None, "groups", False, False),
            ("", "group", False, False),
            ([], "groups", True, False),
            ([], "groups", False, False),
            ("invalid", "groups", False, False),
            ("all", "invalid", False, False),
            ("invalid", "groups", False, True),
            ("invalid", "groups", True, False),
            ("invalid", "groups", True, True),
            ("mean", "summary", True, True),
            ("all", "summary", True, True),
            ("num_inst", "features", True, True),
            ("all", "features", True, True),
        ],
    )
    def test_error_process_generic_set_1(
        self, values, group_name, allow_none, allow_empty
    ):
        with pytest.raises(ValueError):
            _internal.process_generic_set(
                values=values,
                group_name=group_name,
                allow_none=allow_none,
                allow_empty=allow_empty,
            )

    def test_error_process_generic_set_2(self):
        with pytest.raises(TypeError):
            _internal.process_generic_set(values=[1, 2, 3], group_name=None)

    @pytest.mark.parametrize(
        "summary",
        [
            "meanmean",
            "invalid",
        ],
    )
    def test_error_unknown_summary(self, summary):
        with pytest.raises(ValueError):
            MFE(summary=summary)

    @pytest.mark.parametrize(
        "features",
        [
            None,
            [],
            "",
        ],
    )
    def test_error_invalid_features(self, features):
        with pytest.raises(ValueError):
            MFE(features=features)

    @pytest.mark.parametrize(
        "score",
        [
            None,
            [],
            "",
            "invalid",
            "accuracyaccuracy",
        ],
    )
    def test_error_invalid_score(self, score):
        with pytest.raises(ValueError):
            MFE(score=score)

    @pytest.mark.parametrize(
        "rescale",
        [
            "",
            "invalid",
            "minmax",
        ],
    )
    def test_error_invalid_rescale_1(self, rescale):
        with pytest.raises(ValueError):
            X, y = load_xy(0)
            MFE().fit(X=X.values, y=y.values, rescale=rescale)

    def test_error_invalid_rescale_2(self):
        with pytest.raises(TypeError):
            X, y = load_xy(0)
            MFE().fit(X=X.values, y=y.values, rescale=[])

    @pytest.mark.parametrize(
        "features, groups",
        [
            ("invalid", "all"),
            ("invalid", "general"),
            ("mean", "info-theory"),
            ("nr_instt", "general"),
        ],
    )
    def test_warning_invalid_features(self, features, groups):
        with pytest.warns(UserWarning):
            X, y = load_xy(0)
            model = MFE(features=features, groups=groups).fit(
                X=X.values, y=y.values
            )
            model.extract()

    @pytest.mark.parametrize(
        "groups, precomp_groups",
        [
            ("all", "invalid"),
            ("general", "statistical"),
            ("info-theory", "general"),
            (["general", "statistical"], ["general", "info-theory"]),
        ],
    )
    def test_warning_invalid_precomp(self, groups, precomp_groups):
        with pytest.warns(UserWarning):
            X, y = load_xy(0)
            MFE(groups=groups).fit(
                X=X.values, y=y.values, precomp_groups=precomp_groups
            )

    def test_warning_invalid_argument(self):
        with pytest.warns(UserWarning):
            X, y = load_xy(0)
            model = MFE(features="sd").fit(X=X.values, y=y.values)
            model.extract(sd={"ddof": 1, "invalid": "value?"})

    def test_error_rescale_data(self):
        X, y = load_xy(0)
        with pytest.raises(ValueError):
            _internal.rescale_data(X, option="42")

    def test_error_transform_num(self):
        X, y = load_xy(0)
        with pytest.raises(TypeError):
            _internal.transform_num(X, num_bins="")

        with pytest.raises(ValueError):
            _internal.transform_num(X, num_bins=-1)

    def test_isnumeric_check(self):
        assert _internal.isnumeric([]) is False

    def test_error_check_data(self):
        X, y = load_xy(0)
        with pytest.raises(TypeError):
            _internal.check_data(X, y="")

    def test_errors__fill_col_ind_by_type(self):
        X, y = load_xy(0)
        with pytest.raises(TypeError):
            mfe = MFE()
            mfe._fill_col_ind_by_type()

        X = [[1, 2, "a", "b"]] * 10 + [[3, 4, "c", "d"]] * 10
        y = [0] * 10 + [1] * 10

        mfe = MFE()
        mfe.X, mfe.y = np.array(X), np.array(y)
        mfe._fill_col_ind_by_type(cat_cols=None)
        assert mfe._attr_indexes_cat == ()

        mfe = MFE()
        mfe.X, mfe.y = np.array(X), np.array(y)
        mfe._fill_col_ind_by_type(cat_cols="auto", check_bool=True)
        assert len(mfe._attr_indexes_cat) == 4

        mfe = MFE()
        mfe.X, mfe.y = np.array(X), np.array(y)
        mfe._fill_col_ind_by_type(cat_cols=[2, 3])
        assert mfe._attr_indexes_cat == (2, 3)

    def test_error__set_data_categoric(self):
        with pytest.raises(TypeError):
            mfe = MFE()
            mfe._set_data_categoric(True)

        with pytest.raises(TypeError):
            mfe = MFE()
            mfe.X = np.array([])
            mfe._set_data_categoric(True)

    def test_error__set_data_numeric(self):
        with pytest.raises(TypeError):
            mfe = MFE()
            mfe._set_data_numeric(True)

        with pytest.raises(TypeError):
            mfe = MFE()
            mfe.X = np.array([])
            mfe._set_data_numeric(True)

    def test_invalid_cat_transf(self):
        X, y = load_xy(0)
        with pytest.raises(ValueError):
            mfe = MFE()
            mfe.fit(X.values, y.values, transform_cat="invalid")

    def test_extract_with_confidence_without_data(self):
        mfe = MFE()
        with pytest.raises(TypeError):
            mfe.extract_with_confidence()

    def test_bootstrap_extractor_extract_with_confidence_without_data(self):
        mfe = MFE()
        bootstrap_extractor = _bootstrap.BootstrapExtractor(extractor=mfe)
        with pytest.raises(TypeError):
            mfe.extract_with_confidence()
