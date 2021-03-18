"""Test module for General class metafeatures."""
import typing as t
import pytest

import pymfe._internal
import pymfe._summary
import pymfe.mfe
import numpy as np


def test_get_summary():
    assert not set(pymfe.mfe.MFE.valid_summary()).symmetric_difference(
        pymfe._internal.VALID_SUMMARY
    )


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
    assert np.isnan(pymfe._summary.sum_nanstd(sing_val, ddof=1))
    assert np.isnan(pymfe._summary.sum_nanvar(sing_val, ddof=1))
    assert np.isnan(pymfe._summary.sum_nanstd(sing_val, ddof=2))
    assert np.isnan(pymfe._summary.sum_nanvar(sing_val, ddof=2))


@pytest.mark.parametrize(
    "summary_func",
    [
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
    ],
)
def test_nansummary(summary_func):
    values = np.array(
        [
            1,
            np.nan,
            np.nan,
            2,
            -4,
            np.nan,
            9,
            -11,
            1,
            5,
            6.4,
            2.3,
            4.5,
            np.nan,
            0,
        ]
    )
    clean_values = values[~np.isnan(values)]
    summary_nan = pymfe._summary.SUMMARY_METHODS[summary_func]
    summary_reg = pymfe._summary.SUMMARY_METHODS[summary_func[3:]]

    assert np.allclose(
        summary_nan(list(values)), summary_reg(list(clean_values))
    )


def test_nancount():
    values = np.array(
        [
            1,
            np.nan,
            np.nan,
            2,
            -4,
            np.nan,
            9,
            -11,
            1,
            5,
            6.4,
            2.3,
            4.5,
            np.nan,
            0,
        ]
    )
    summary_nan = pymfe._summary.SUMMARY_METHODS["nancount"]
    summary_reg = pymfe._summary.SUMMARY_METHODS["count"]
    assert np.allclose(
        summary_nan(list(values)),
        summary_reg(list(values)) - np.count_nonzero(np.isnan(values)),
    )


@pytest.mark.parametrize("p", [-1, 0, 1, 2, 3, 4])
def test_powersum_scalar(p: t.Union[int, float]):
    values = [0, 0, -1, 10, -10, -5, 8, 2.5, 0.1, -0.2]

    res_a = pymfe._summary.sum_powersum(values, p)
    res_b = np.sum(np.power(values, p))

    assert np.isclose(res_a, res_b)


@pytest.mark.parametrize("p", [-1, 0, 1, 2, 3, 4])
def test_nanpowersum_scalar(p: t.Union[int, float]):
    values = [0, np.nan, -1, np.nan, -10, -5, 8, 2.5, 0.1, -0.2, np.nan]

    res_a = pymfe._summary.sum_nanpowersum(values, p)
    res_b = np.nansum(np.power(pymfe._summary._remove_nan(values), p))

    assert np.isclose(res_a, res_b)


@pytest.mark.parametrize("p", [[2], [-1, 0], [1, 2, 3, 4]])
def test_powersum_array(p: t.Sequence[t.Union[int, float]]):
    values = [0, 0, -1, 10, -10, -5, 8, 2.5, 0.1, -0.2]

    res_a = pymfe._summary.sum_powersum(values, p)
    res_b = [np.sum(np.power(values, cur_p)) for cur_p in p]

    assert len(res_a) == len(p) and np.allclose(res_a, res_b)


@pytest.mark.parametrize("p", [[2], [-1, 0], [1, 2, 3, 4]])
def test_nanpowersum_array(p: t.Sequence[t.Union[int, float]]):
    values = [0, np.nan, -1, np.nan, -10, -5, 8, 2.5, 0.1, -0.2, np.nan]

    res_a = pymfe._summary.sum_nanpowersum(values, p)
    res_b = [
        np.nansum(np.power(pymfe._summary._remove_nan(values), cur_p))
        for cur_p in p
    ]

    assert len(res_a) == len(p) and np.allclose(res_a, res_b)


@pytest.mark.parametrize("p", [-1, 0, 1, 2, 3, 4])
def test_pnorm_scalar(p: t.Union[int, float]):
    values = [0, 0, -1, 10, -10, -5, 8, 2.5, 0.1, -0.2]

    res_a = pymfe._summary.sum_pnorm(values, p)
    res_b = np.linalg.norm(values, p) if p >= 0 else np.nan

    assert np.isclose(res_a, res_b, equal_nan=True)


@pytest.mark.parametrize("p", [-1, 0, 1, 2, 3, 4])
def test_nanpnorm_scalar(p: t.Union[int, float]):
    values = [0, np.nan, -1, np.nan, -10, -5, 8, 2.5, 0.1, -0.2, np.nan]

    res_a = pymfe._summary.sum_nanpnorm(values, p)
    res_b = (
        np.linalg.norm(pymfe._summary._remove_nan(values), p)
        if p >= 0
        else np.nan
    )

    assert np.isclose(res_a, res_b, equal_nan=True)


@pytest.mark.parametrize("p", [[2], [-1, 0], [1, 2, 3, 4]])
def test_pnorm_array(p: t.Sequence[t.Union[int, float]]):
    values = [0, 0, -1, 10, -10, -5, 8, 2.5, 0.1, -0.2]

    res_a = pymfe._summary.sum_pnorm(values, p)
    res_b = [
        np.linalg.norm(values, cur_p) if cur_p >= 0 else np.nan for cur_p in p
    ]

    assert len(res_a) == len(p) and np.allclose(res_a, res_b, equal_nan=True)


@pytest.mark.parametrize("p", [[2], [-1, 0], [1, 2, 3, 4]])
def test_nanpnorm_array(p: t.Sequence[t.Union[int, float]]):
    values = [0, np.nan, -1, np.nan, -10, -5, 8, 2.5, 0.1, -0.2, np.nan]

    res_a = pymfe._summary.sum_nanpnorm(values, p)
    res_b = [
        np.linalg.norm(pymfe._summary._remove_nan(values), cur_p)
        if cur_p >= 0
        else np.nan
        for cur_p in p
    ]

    assert len(res_a) == len(p) and np.allclose(res_a, res_b, equal_nan=True)


def test_sum_sum():
    values = [0, 0, -1, 10, -10, -5, 8, 2.5, 0.1, -0.2]
    assert np.isclose(sum(values), pymfe._summary.sum_sum(values))


def test_sum_nansum():
    values = [0, np.nan, -1, np.nan, -10, -5, 8, 2.5, 0.1, -0.2, np.nan]
    assert np.isclose(np.nansum(values), pymfe._summary.sum_nansum(values))


@pytest.mark.parametrize(
    "summary, sum_args, exp_len",
    (
        ("mean", None, 1),
        ("nanmean", None, 1),
        ("sd", None, 1),
        ("nansd", None, 1),
        ("var", None, 1),
        ("nanvar", None, 1),
        ("histogram", {"bins": 7}, 7),
        ("nanhistogram", {"bins": 7}, 7),
        ("iq_range", None, 1),
        ("naniq_range", None, 1),
        ("kurtosis", None, 1),
        ("nankurtosis", None, 1),
        ("max", None, 1),
        ("nanmax", None, 1),
        ("median", None, 1),
        ("nanmedian", None, 1),
        ("min", None, 1),
        ("nanmin", None, 1),
        ("quantiles", None, 5),
        ("nanquantiles", None, 5),
        ("range", None, 1),
        ("nanrange", None, 1),
        ("skewness", None, 1),
        ("nanskewness", None, 1),
        ("sum", None, 1),
        ("nansum", None, 1),
        ("powersum", None, 1),
        ("powersum", {"p": [-1, 0, 1, 2]}, 4),
        ("pnorm", None, 1),
        ("pnorm", {"p": [-1, 0, 1, 2]}, 4),
        ("nanpowersum", None, 1),
        ("nanpowersum", {"p": [-1, 0, 1, 2]}, 4),
        ("nanpnorm", None, 1),
        ("nanpnorm", {"p": [-1, 0, 1, 2]}, 4),
    ),
)
def test_summary_empty_slice(
    summary: str, sum_args: t.Dict[str, t.Any], exp_len: int
):
    if sum_args is None:
        sum_args = {}

    X = np.asarray([1, 2, 3], dtype=str)

    extractor = pymfe.mfe.MFE(features="mean", summary=summary).fit(
        X, transform_cat=None
    )

    res = extractor.extract(suppress_warnings=True, **{summary: sum_args})[1]

    assert len(res) == exp_len and np.all(np.isnan(res))


@pytest.mark.parametrize(
    "summary, sum_args, exp_len",
    (
        ("mean", None, 1),
        ("nanmean", None, 1),
        ("sd", None, 1),
        ("nansd", None, 1),
        ("var", None, 1),
        ("nanvar", None, 1),
        ("histogram", {"bins": 7}, 7),
        ("nanhistogram", {"bins": 7}, 7),
        ("iq_range", None, 1),
        ("naniq_range", None, 1),
        ("kurtosis", None, 1),
        ("nankurtosis", None, 1),
        ("max", None, 1),
        ("nanmax", None, 1),
        ("median", None, 1),
        ("nanmedian", None, 1),
        ("min", None, 1),
        ("nanmin", None, 1),
        ("quantiles", None, 5),
        ("nanquantiles", None, 5),
        ("range", None, 1),
        ("nanrange", None, 1),
        ("skewness", None, 1),
        ("nanskewness", None, 1),
        ("sum", None, 1),
        ("nansum", None, 1),
        ("powersum", None, 1),
        ("powersum", {"p": [-1, 0, 1, 2]}, 4),
        ("pnorm", None, 1),
        ("pnorm", {"p": [-1, 0, 1, 2]}, 4),
        ("nanpowersum", None, 1),
        ("nanpowersum", {"p": [-1, 0, 1, 2]}, 4),
        ("nanpnorm", None, 1),
        ("nanpnorm", {"p": [-1, 0, 1, 2]}, 4),
    ),
)
def test_summary_all_nan(
    summary: str, sum_args: t.Dict[str, t.Any], exp_len: int
):
    if sum_args is None:
        sum_args = {}

    X = np.full(5, fill_value=np.nan)

    extractor = pymfe.mfe.MFE(features="mean", summary=summary)
    extractor.fit(X, transform_cat=None, transform_num=None)

    res = extractor.extract(suppress_warnings=True, **{summary: sum_args})[1]

    assert len(res) == exp_len and np.all(np.isnan(res))
