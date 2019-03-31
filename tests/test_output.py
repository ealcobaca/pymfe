"""Test module for MFE class output details."""
import pytest

from pymfe.mfe import MFE
from tests.utils import load_xy

GNAME = "mfe-output-details"


class TestErrorsWarnings:
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
            ])
        def test_output_lengths_2(self, dt_id, measure_time):
            X, y = load_xy(dt_id)
            res = MFE(measure_time=measure_time).fit(X=X.values,
                                                     y=y.values).extract()
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
            ])
        def test_output_lengths_4(self, dt_id, measure_time):
            X, y = load_xy(dt_id)
            res = MFE(summary=None,
                      measure_time=measure_time).fit(X=X.values,
                                                     y=y.values).extract()
            vals, names, time = res

            assert len(vals) == len(names) == len(time)
