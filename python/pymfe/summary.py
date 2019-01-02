"""Module dedicated for functions that summarizes feature values.

Attributes:
    SUMMARY_METHODS:
"""
import typing as tp

from scipy import stats
import numpy as np

TypeValList = tp.Iterable[tp.TypeVar("Numeric", int, float)]
"""."""

TypeNpArray = tp.TypeVar("TypeNpArray", np.array)
"""."""


def sum_histogram(
        values: TypeValList,
        bins: int = 5,
        normalize: bool = False) -> TypeNpArray[float]:
    """Todo."""

    values, _ = np.histogram(bins=bins, density=normalize)

    return values


def sum_quartiles(values: TypeValList) -> TypeNpArray[float]:
    """."""
    return np.percentile(values, (0, 25, 50, 75, 100))


SUMMARY_METHODS = {
    "mean": np.mean,
    "sd": np.std,
    "count": len,
    "histogram": sum_histogram,
    "iq_range": stats.iqr,
    "kurtosis": stats.kurtosis,
    "max": max,
    "median": np.median,
    "min": min,
    "quartiles": sum_quartiles,
    "range": np.ptp,
    "skewness": stats.skew,
}
