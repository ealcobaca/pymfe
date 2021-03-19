"""Test module for MFE class output details."""
import pytest
import sklearn.tree
import numpy as np
import pandas as pd

import pymfe._internal as _internal
from pymfe.mfe import MFE
from tests.utils import load_xy

GNAME = "mfe-output-details"


class TestOutput:
    """TestClass dedicated to test MFE output details."""

    def test_output_lengths_1(self):
        X, y = load_xy(0)
        res = MFE().fit(X=X.values, y=y.values).extract()
        vals, names = res

        assert len(vals) == len(names)

    @pytest.mark.parametrize(
        "dt_id, measure_time",
        [
            (0, "total"),
            (0, "total_summ"),
            (0, "avg"),
            (0, "avg_summ"),
            (2, "total"),
            (2, "total_summ"),
            (2, "avg"),
            (2, "avg_summ"),
        ],
    )
    def test_output_lengths_2(self, dt_id, measure_time):
        X, y = load_xy(dt_id)
        res = (
            MFE(measure_time=measure_time)
            .fit(X=X.values, y=y.values)
            .extract()
        )
        vals, names, time = res

        assert len(vals) == len(names) == len(time)

    def test_output_lengths_3(self):
        X, y = load_xy(0)
        res = MFE(summary=None).fit(X=X.values, y=y.values).extract()
        vals, names = res

        assert len(vals) == len(names)

    @pytest.mark.parametrize(
        "dt_id, measure_time",
        [
            (0, "total"),
            (0, "total_summ"),
            (0, "avg"),
            (0, "avg_summ"),
            (2, "total"),
            (2, "total_summ"),
            (2, "avg"),
            (2, "avg_summ"),
        ],
    )
    def test_output_lengths_4(self, dt_id, measure_time):
        X, y = load_xy(dt_id)
        res = (
            MFE(summary=None, measure_time=measure_time)
            .fit(X=X.values, y=y.values)
            .extract()
        )
        vals, names, time = res

        assert len(vals) == len(names) == len(time)

    def test_verbosity_2(self, capsys):
        X, y = load_xy(0)

        MFE().fit(X=X.values, y=y.values).extract(verbose=0)

        captured = capsys.readouterr().out

        assert not captured

    @pytest.mark.parametrize(
        "verbosity, msg_expected",
        [
            (0, False),
            (1, True),
            (2, True),
        ],
    )
    def test_verbosity_3(self, verbosity, msg_expected, capsys):
        X, y = load_xy(0)

        MFE().fit(X=X.values, y=y.values).extract(verbose=verbosity)

        captured = capsys.readouterr().out
        assert (not msg_expected) or captured

    @pytest.mark.parametrize(
        "verbosity, msg_expected",
        [
            (0, False),
            (1, True),
        ],
    )
    def test_verbosity_with_confidence(self, verbosity, msg_expected, capsys):
        X, y = load_xy(2)

        MFE().fit(X.values, y.values).extract_with_confidence(
            verbose=verbosity
        )

        captured = capsys.readouterr().out
        assert ((not msg_expected) and (not captured)) or (
            msg_expected and captured
        )

    @pytest.mark.parametrize(
        "verbosity, msg_expected",
        [
            (0, False),
            (1, True),
        ],
    )
    def test_verbosity_from_model(self, verbosity, msg_expected, capsys):
        X, y = load_xy(2)

        model = sklearn.tree.DecisionTreeClassifier().fit(X.values, y.values)

        MFE().extract_from_model(model, verbose=verbosity)

        captured = capsys.readouterr().out
        assert ((not msg_expected) and (not captured)) or (
            msg_expected and captured
        )

    def test_extract_output_default(self):
        X, y = load_xy(2)
        extractor = MFE(groups="general").fit(X.values, y.values)
        res = extractor.extract()
        assert isinstance(res, tuple)
        assert len(res) == 2

    def test_extract_output_default_unsupervised(self):
        X, _ = load_xy(2)
        extractor = MFE(groups="general").fit(X.values)
        res = extractor.extract()
        assert isinstance(res, tuple)
        assert len(res) == 2

    def test_extract_output_tuple(self):
        X, y = load_xy(2)
        extractor = MFE(groups="general").fit(X.values, y.values)
        res = extractor.extract(out_type=tuple)
        assert isinstance(res, tuple)
        assert len(res) == 2

    def test_extract_output_dictionary(self):
        X, y = load_xy(2)
        extractor = MFE(groups="general").fit(X.values, y.values)
        res = extractor.extract(out_type=dict)
        assert isinstance(res, dict)
        assert len(res) == 2

    def test_extract_with_time_output_dictionary(self):
        X, y = load_xy(2)
        extractor = MFE(groups="general", measure_time="total").fit(
            X.values, y.values
        )
        res = extractor.extract(out_type=dict)
        assert isinstance(res, dict)
        assert len(res) == 3

    def test_extract_output_dictionary_unsupervised(self):
        X, _ = load_xy(2)
        extractor = MFE(groups="general").fit(X.values)
        res = extractor.extract(out_type=dict)
        assert isinstance(res, dict)
        assert len(res) == 2

    def test_extract_with_confidence_output_dictionary(self):
        X, y = load_xy(2)
        extractor = MFE(groups="general").fit(X.values, y.values)
        res = extractor.extract_with_confidence(
            3, arguments_extract=dict(out_type=dict)
        )
        assert isinstance(res, dict)
        assert len(res) == 3

    def test_extract_with_time_and_with_confidence_output_dictionary(self):
        X, y = load_xy(2)
        extractor = MFE(groups="general", measure_time="total").fit(
            X.values, y.values
        )
        res = extractor.extract_with_confidence(
            3, arguments_extract=dict(out_type=dict)
        )
        assert isinstance(res, dict)
        assert len(res) == 4

    def test_extract_with_confidence_output_dictionary_unsupervised(self):
        X, _ = load_xy(2)
        extractor = MFE(groups="general").fit(X.values)
        res = extractor.extract_with_confidence(
            3, arguments_extract=dict(out_type=dict)
        )
        assert isinstance(res, dict)
        assert len(res) == 3

    def test_extract_output_pandas_dataframe(self):
        X, y = load_xy(2)
        extractor = MFE(groups="general").fit(X.values, y.values)
        expected_mtfs = extractor.extract_metafeature_names()
        res = extractor.extract(out_type=pd.DataFrame)
        assert isinstance(res, pd.DataFrame)
        assert res.values.shape == (1, len(expected_mtfs)) and np.array_equal(
            res.columns, expected_mtfs
        )

    def test_extract_output_pandas_dataframe_unsupervised(self):
        X, _ = load_xy(2)
        extractor = MFE(groups="general").fit(X.values)
        expected_mtfs = extractor.extract_metafeature_names()
        res = extractor.extract(out_type=pd.DataFrame)
        assert isinstance(res, pd.DataFrame)
        assert res.values.shape == (1, len(expected_mtfs)) and np.array_equal(
            res.columns, expected_mtfs
        )

    def test_extract_with_time_output_pandas_dataframe(self):
        X, y = load_xy(2)
        extractor = MFE(measure_time="total", groups="general").fit(
            X.values, y.values
        )
        expected_mtfs = extractor.extract_metafeature_names()
        res = extractor.extract(out_type=pd.DataFrame)
        assert isinstance(res, pd.DataFrame)
        assert res.values.shape == (2, len(expected_mtfs)) and np.array_equal(
            res.columns, expected_mtfs
        )

    def test_extract_with_time_output_pandas_dataframe_unsupervised(self):
        X, _ = load_xy(2)
        extractor = MFE(measure_time="total", groups="general").fit(X.values)
        expected_mtfs = extractor.extract_metafeature_names()
        res = extractor.extract(out_type=pd.DataFrame)
        assert isinstance(res, pd.DataFrame)
        assert res.values.shape == (2, len(expected_mtfs)) and np.array_equal(
            res.columns, expected_mtfs
        )

    def test_invalid_output_type(self):
        X, y = load_xy(2)
        extractor = MFE(groups="general").fit(X.values, y.values)
        with pytest.raises(TypeError):
            res = extractor.extract(out_type=set)

    def test_invalid_output_type_unsupervised(self):
        X, _ = load_xy(2)
        extractor = MFE(groups="general").fit(X.values)
        with pytest.raises(TypeError):
            res = extractor.extract(out_type=set)

    def test_invalid_output_type_with_confidence(self):
        X, y = load_xy(2)
        extractor = MFE(groups="general").fit(X.values, y.values)
        with pytest.raises(TypeError):
            res = extractor.extract_with_confidence(
                3, arguments_extract=dict(out_type=set)
            )

    def test_invalid_output_type_with_confidence_unsupervised(self):
        X, _ = load_xy(2)
        extractor = MFE(groups="general").fit(X.values)
        with pytest.raises(TypeError):
            res = extractor.extract_with_confidence(
                3, arguments_extract=dict(out_type=set)
            )
