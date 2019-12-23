"""Test module for General class metafeatures."""
import pytest

import pymfe._internal
import pymfe._summary
import pymfe.mfe
import numpy as np


def test_get_summary():
    assert (not set(pymfe.mfe.MFE.valid_summary()).symmetric_difference(
        pymfe._internal.VALID_SUMMARY))


def test_sum_histogram():
    mf = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    aux = pymfe._summary.sum_histogram(mf, bins=5)
    assert np.allclose(np.array([0.2, 0.2, 0.2, 0.2, 0.2]), aux)


def test_sum_quantiles():
    mf = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    aux = pymfe._summary.sum_quantiles(mf, package="numpy")
    assert np.allclose(np.array([1.0, 3.25, 5.5, 7.75, 10.0]), aux)

    with pytest.raises(ValueError):
        pymfe._summary.sum_quantiles(mf, package="asd")

    aux = pymfe._summary.sum_quantiles(mf, package="scipy")
    assert np.allclose(np.array([1.0, 2.95, 5.5, 8.05, 10.0]), aux)


def test_sum_skewness():
    mf = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    aux = pymfe._summary.sum_skewness(mf)
    assert np.allclose(0.0, aux)

    with pytest.raises(ValueError):
        pymfe._summary.sum_skewness(mf, method=4)

    aux = pymfe._summary.sum_skewness([])
    assert aux is np.nan

    mf = [10.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    aux = pymfe._summary.sum_skewness(mf, method=2)
    assert np.allclose(-0.15146310708295876, aux)


def test_sum_kurtosis():
    mf = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    aux = pymfe._summary.sum_kurtosis(mf)
    assert np.allclose(-1.5616363636363637, aux)

    with pytest.raises(ValueError):
        pymfe._summary.sum_kurtosis(mf, method=4)

    aux = pymfe._summary.sum_kurtosis([])
    assert aux is np.nan

    mf = [10.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    aux = pymfe._summary.sum_kurtosis(mf, method=2)
    assert np.allclose(-1.356984911550468, aux)


def test_ddof():
    sing_val = [1.0]
    assert np.isclose(0.0, pymfe._summary.sum_std(sing_val, ddof=0))
    assert np.isclose(0.0, pymfe._summary.sum_var(sing_val, ddof=0))
    assert np.isnan(pymfe._summary.sum_std(sing_val, ddof=1))
    assert np.isnan(pymfe._summary.sum_var(sing_val, ddof=1))
    assert np.isnan(pymfe._summary.sum_std(sing_val, ddof=2))
    assert np.isnan(pymfe._summary.sum_var(sing_val, ddof=2))


@pytest.mark.parametrize("summary_func", [
    "nanmean",
    "nansd",
    "nanvar",
    "nanhistogram",
    "naniq_range",
    "nankurtosis",
    "nanmax",
    "nanmedian",
    "nanmin",
    "nanquantiles",
    "nanrange",
    "nanskewness",
])
def test_nansummary(summary_func):
    values = np.array([
        1, np.nan, np.nan, 2, -4, np.nan, 9, -11, 1, 5, 6.4, 2.3, 4.5, np.nan,
        0
    ])
    clean_values = values[~np.isnan(values)]
    summary_nan = pymfe._summary.SUMMARY_METHODS[summary_func]
    summary_reg = pymfe._summary.SUMMARY_METHODS[summary_func[3:]]

    assert np.allclose(summary_nan(values), summary_reg(clean_values))


def test_nancount():
    values = np.array([
        1, np.nan, np.nan, 2, -4, np.nan, 9, -11, 1, 5, 6.4, 2.3, 4.5, np.nan,
        0,
    ])
    summary_nan = pymfe._summary.SUMMARY_METHODS["nancount"]
    summary_reg = pymfe._summary.SUMMARY_METHODS["count"]
    assert np.allclose(
        summary_nan(values),
        summary_reg(values) - np.count_nonzero(np.isnan(values)))
