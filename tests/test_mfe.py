"""Test module for core class MFE."""
import pytest

import context  # noqa: F401
from pymfe.mfe import MFE


class TestMFE:
    """TestClass dedicated to test core class MFE."""

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
        mfe_groups = set(MFE._process_groups(groups))
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
        mfe_groups = set(MFE._process_groups(groups))
        assert not mfe_groups.difference(expected)

    DATA_GROUPS_ITERABLE_NOTVALID = (
        (["unknown"], ValueError),
        (["statistical", "basic"], ValueError),
        (["MODEL-BASED", "info_theory"], ValueError),
        (("landMARKING", "statisticall"), ValueError),
        ([""], ValueError),
        ([], AttributeError),
    )

    @pytest.mark.parametrize("groups, expected_exception",
                             DATA_GROUPS_ITERABLE_NOTVALID)
    def test_param_groups_iterable_notvalid(self, groups, expected_exception):
        """Tests 'group' param (_process_groups), invalid iterable input."""
        with pytest.raises(expected_exception):
            set(MFE._process_groups(groups))

    DATA_GROUPS_SINGLE_NOTVALID = (
        ("unknown", ValueError),
        ("", AttributeError),
        (" ", ValueError),
        ("info_theory", ValueError),
        ("model based", ValueError),
    )

    @pytest.mark.parametrize("groups, expected_exception",
                             DATA_GROUPS_SINGLE_NOTVALID)
    def test_param_groups_single_notvalid(self, groups, expected_exception):
        """Tests 'group' param (_process_groups), invalid single-val input."""
        with pytest.raises(expected_exception):
            set(MFE._process_groups(groups))
