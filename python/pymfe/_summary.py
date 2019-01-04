"""Module dedicated for functions that summarizes feature values.

Attributes:
    SUMMARY_METHODS (:obj:`Dict`): dictionary that links summary
        function names as keys with methods Callables which imple-
        ments then as values.
"""
import typing as t

from scipy import stats
import numpy as np

TypeNumeric = t.TypeVar("TypeNumeric", int, float)
"""Type annotation for a Numeric type (int or float)."""

TypeValList = t.Iterable[TypeNumeric]
"""Type annotation for a Iterable of a Numeric type (int or float)."""


def sum_histogram(values: TypeValList, bins: int = 5,
                  normalize: bool = False) -> TypeValList:
    """Returns a list of frequencies/density of a histogram of given values.

    Args:
        values: collection of values which histogram is made from.
        bins: number of bins (separations) of the histogram.
        normalize: if True, returns density values instead of fre-
            quencies.
    """

    values, _ = np.histogram(values, bins=bins, density=normalize)

    return values


def sum_quartiles(values: TypeValList) -> TypeValList:
    """Calc. min, first quartile, median, third quartile and max of values."""
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
