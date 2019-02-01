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

TypeValList = t.Sequence[TypeNumeric]
"""Type annotation for a Sequence of a Numeric type (int or float)."""


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


def skewness(values: TypeValList, method=3, bias=True) -> float:
    """Calculate skewness from data using ``method`` strategy.

    Args:
        values (:obj:`list` or numeric): values from where skewness is
            estimated.

        method (:obj:`int`, optional): defines the strategy used for
            estimate data skewness. Used for total compatibility with
            R package ``e1071``. The options must be one of the follo-
            wing:

            Option      Formula
            -------------------
            1           Skew_1 = m_3 / m_2^(3/2) (default of ``scipy.stats``)
            2           Skew_2 = Skew_1 * sqrt(n(n-1)) / (n-2)
            3           Skew_3 = m_3 / s^3 = Skew_1 ((n-1)/n)^(3/2)

            Where ``n`` is the number of elements in ``values`` and
            m_i is the ith momentum of ``values``.

            Note that if the selected method is unable to be calculated due
            to division by zero, then the first method will be used instead.

        bias (:obj:`bool`, optional): If False, then the calculations
            are corrected for statistical bias.

    Return:
        float: estimated kurtosis from ``values`` using ``method``.
    """
    if method not in (1, 2, 3):
        raise ValueError('Invalid method "{}" for'
                         "extracting skewness".format(method))

    num_vals = len(values)

    if num_vals == 0:
        return np.nan

    skew_val = scipy.stats.skew(values, bias=bias)

    if method == 2 and num_vals != 2:
        skew_val *= (num_vals*(num_vals - 1.0))**0.5 / (num_vals - 2.0)

    elif method == 3:
        skew_val *= ((num_vals - 1.0) / num_vals)**(1.5)

    return skew_val


def kurtosis(values: TypeValList, method=3, bias=True) -> TypeValList:
    """
    Args:
        values (:obj:`list` of numeric): values from where kurtosis is
            estimated.

        method (:obj:`int`, optional): defines the strategy used for
            estimate data kurtosis. Used for total compatibility with
            R package ``e1071``. The options must be one of the follo-
            wing:

            Option      Formula
            -------------------
            1           Kurt_1 = m_4 / m_2^2 - 3. (default of ``scipy.stats``)
            2           Kurt_2 = ((n+1) * Kurt_1 + 6) * (n-1) / ((n-2)*(n-3)).
            3           Kurt_3 = m_4 / s^4 - 3 = (Kurt_1+3) * (1 - 1/n)^2 - 3.

            Where ``n`` is the number of elements in ``values`` and
            m_i is the ith momentum of ``values``.

            Note that if the selected method is unable to be calculated due
            to division by zero, then the first method will be used instead.

        bias (:obj:`bool`, optional): If False, then the calculations
            are corrected for statistical bias.
    """
    if method not in (1, 2, 3):
        raise ValueError('Invalid method "{}" for'
                         "extracting kurtosis".format(method))

    num_vals = len(values)

    if num_vals == 0:
        return np.nan

    kurt_val = scipy.stats.kurtosis(values, bias=bias)

    if method == 2 and num_vals > 3:
        kurt_val = (num_vals + 1.0) * kurt_val + 6
        kurt_val *= (num_vals - 1.0) / ((num_vals-2.0) * (num_vals-3.0))

    elif method == 3:
        kurt_val = (kurt_val + 3.0) * (1.0 - 1.0/num_vals)**2.0 - 3.0

    return kurt_val


SUMMARY_METHODS = {
    "mean": np.mean,
    "sd": np.std,
    "var": np.var,
    "count": len,
    "histogram": sum_histogram,
    "iq_range": scipy.stats.iqr,
    "kurtosis": kurtosis,
    "max": max,
    "median": np.median,
    "min": min,
    "quantiles": sum_quantiles,
    "range": np.ptp,
    "skewness": skewness,
}
