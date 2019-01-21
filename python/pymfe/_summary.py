"""Module dedicated for auxiliary functions for feature summarization.

Attributes:
    SUMMARY_METHODS (:obj:`Dict`): dictionary that links summary
        function names as keys with methods Callables which imple-
        ments then as values.
"""
import typing as t

import scipy.stats
import numpy as np

TypeNumeric = t.TypeVar("TypeNumeric", int, float)
"""Type annotation for a Numeric type (int or float)."""

TypeValList = t.Iterable[TypeNumeric]
"""Type annotation for a Iterable of a Numeric type (int or float)."""


def sum_histogram(values: TypeValList, bins: int = 10,
                  normalize: bool = True) -> TypeValList:
    """Returns a list of abs/rel. frequencies of a histogram of given values.

    Args:
        values (:obj:`List` of numeric): collection of values which histo-
            gram will be made from.

        bins (:obj:`int`, optional): number of bins (separations) of the
            histogram.

        normalize(:obj:`bool`, optional): if True, returns density values
            instead of frequencies.

    Returns:
        list: frequencies or density of each bin.

    Raises:
        TypeError: if `values` contains non-numeric data.
    """

    freqs, _ = np.histogram(values, bins=bins)

    if normalize:
        freqs = freqs / sum(freqs)

    return freqs


def sum_quantiles(values: TypeValList) -> TypeValList:
    """Calc. min, first quartile, median, third quartile and max of values.

    Args:
        values (:obj:`List` of numeric): values to quartiles be calculated
            from.
    """
    return np.percentile(values, (0, 25, 50, 75, 100))


SUMMARY_METHODS = {
    "mean": np.mean,
    "sd": np.std,
    "var": np.var,
    "count": len,
    "histogram": sum_histogram,
    "iq_range": scipy.stats.iqr,
    "kurtosis": scipy.stats.kurtosis,
    "max": max,
    "median": np.median,
    "min": min,
    "quantiles": sum_quantiles,
    "range": np.ptp,
    "skewness": scipy.stats.skew,
}
