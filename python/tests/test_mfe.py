"""Test module for core class MFE and _internal module."""
import pytest

import context  # noqa: F401
# from pymfe.mfe import MFE
from pymfe import _internal


class TestMFE:
    """TestClass dedicated to test core class MFE and _internal module."""

    ALL_GROUPS = ("landmarking",
                  "general",
                  "statistical",
                  "model-based",
                  "info-theory")

    DATA_GROUPS_ITERABLE_VALID = (
        (["all"], ALL_GROUPS),
        (["ALL"], ALL_GROUPS),
        (("landmarking", "statistical", "all"), ALL_GROUPS),
        ({"landmarking", "info-theory"}, ("landmarking", "info-theory")),
        (("statistical", "general"), ("general", "statistical")),
        (("landMARKING",), ("landmarking",)),
        (("MODEL-based", "Statistical",
          "GENERAL", "info-THEORY", "LaNdMaRKiNg"), ALL_GROUPS),
    )

    @pytest.mark.parametrize("groups, expected", DATA_GROUPS_ITERABLE_VALID)
    def test_param_groups_iterable_valid(self, groups, expected):
        """Tests 'group' param (_process_groups), valid iterable input."""
        mfe_groups = set(_internal.process_groups(groups))
        assert not mfe_groups.difference(expected)

    DATA_GROUPS_SINGLE_VALID = (
        ("info-theory", ("info-theory",)),
        ("statistical", ("statistical",)),
        ("general", ("general",)),
        ("model-based", ("model-based",)),
        ("landmarking", ("landmarking",)),
    )

    @pytest.mark.parametrize("groups, expected", DATA_GROUPS_SINGLE_VALID)
    def test_param_groups_single_valid(self, groups, expected):
        """Tests 'group' param (_process_groups), valid single-valued input."""
        mfe_groups = set(_internal.process_groups(groups))
        assert not mfe_groups.difference(expected)

    DATA_GROUPS_ITERABLE_NOTVALID = (
        (["unknown"], ValueError),
        (["statistical", "basic"], ValueError),
        (["MODEL-BASED", "info_theory"], ValueError),
        (("landMARKING", "statisticall"), ValueError),
        ([""], ValueError),
        ([], ValueError),
        (12, TypeError),
    )

    @pytest.mark.parametrize("groups, expected_exception",
                             DATA_GROUPS_ITERABLE_NOTVALID)
    def test_param_groups_iterable_notvalid(self, groups, expected_exception):
        """Tests 'group' param (_process_groups), invalid iterable input."""
        with pytest.raises(expected_exception):
            set(_internal.process_groups(groups))

    DATA_GROUPS_SINGLE_NOTVALID = (
        ("unknown", ValueError),
        ("", ValueError),
        (None, ValueError),
        (" ", ValueError),
        ("info_theory", ValueError),
        ("model based", ValueError),
        (123, TypeError),
    )

    @pytest.mark.parametrize("groups, expected_exception",
                             DATA_GROUPS_SINGLE_NOTVALID)
    def test_param_groups_single_notvalid(self, groups, expected_exception):
        """Tests 'group' param (_process_groups), invalid single-val input."""
        with pytest.raises(expected_exception):
            set(_internal.process_groups(groups))

    DATA_SUMMARY_SINGLE_VALID = (
        ("mean", ("mean",)),
        ("sd", ("sd",)),
    )

    @pytest.mark.parametrize("summary, expected", DATA_SUMMARY_SINGLE_VALID)
    def test_param_summary_single_valid(self, summary, expected):
        """Tests 'summary' (_process_summary), valid single-valued input."""
        mfe_summary = set(_internal.process_summary(summary))
        assert not mfe_summary.difference(expected)

    DATA_SUMMARY_ITERABLE_VALID = (
        (("mean", "sd"), ("mean", "sd",)),
        (("MeAn", "SD"), ("mean", "sd",)),
    )

    @pytest.mark.parametrize("summary, expected", DATA_SUMMARY_ITERABLE_VALID)
    def test_param_summary_iterable_valid(self, summary, expected):
        """Tests 'summary' (_process_summary), valid iterable input."""
        mfe_summary = set(_internal.process_summary(summary))
        assert not mfe_summary.difference(expected)

    DATA_SUMMARY_SINGLE_NOTVALID = (
        ("unknown", ValueError),
        ("", ValueError),
        (None, ValueError),
        (" ", ValueError),
        (tuple(), ValueError),
        ("meann", ValueError),
        (123, TypeError),
    )

    @pytest.mark.parametrize("summary, expected_exception",
                             DATA_SUMMARY_SINGLE_NOTVALID)
    def test_param_summary_single_notvalid(self, summary, expected_exception):
        """Tests 'summary' (_process_summary), invalid single-val input."""
        with pytest.raises(expected_exception):
            set(_internal.process_summary(summary))

    DATA_SUMMARY_ITERABLE_NOTVALID = (
        (("mean", "sd", "unknown"), ValueError),
        (("sd", ""), ValueError),
        (("mean", None), TypeError),
        (("mean", "sd", 123), TypeError),
    )

    @pytest.mark.parametrize("summary, expected_exception",
                             DATA_SUMMARY_ITERABLE_NOTVALID)
    def test_param_summary_iterable_notvalid(self, summary,
                                             expected_exception):
        """Tests 'summary' (_process_summary), invalid iterable input."""
        with pytest.raises(expected_exception):
            set(_internal.process_summary(summary))
