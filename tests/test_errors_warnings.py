"""Test module for MFE class errors and warnings."""
import pytest

from pymfe.mfe import MFE
from tests.utils import load_xy

GNAME = "errors-warnings"


class TestErrorsWarnings:
    """TestClass dedicated to test General metafeatures."""

    def test_error_empty_data_1(self):
        with pytest.raises(TypeError):
            MFE().fit(X=None, y=None)

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
            ("invalid", ),
        ])
    def test_error_invalid_groups(self, group_name):
        with pytest.raises(ValueError):
            MFE(groups=group_name)

    def test_error_random_state(self):
        with pytest.raises(ValueError):
            MFE(random_state=1.5)

    def test_error_folds(self):
        with pytest.raises(ValueError):
            MFE(folds=1.5)

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

    def test_warning_invalid_feature(self):
        with pytest.warns(UserWarning):
            X, y = load_xy(0)
            model = MFE(features="invalid").fit(X=X.values, y=y.values)
            model.extract()
