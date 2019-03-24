"""A module dedicated for auxiliary functions for feature summarization.

Attributes:
    SUMMARY_METHODS (:obj:`Dict`): dictionary that links summary function
        names as keys with methods callables which implements then as values.
"""
import typing as t
import collections

import scipy.stats
import numpy as np

TypeNumeric = t.TypeVar("TypeNumeric", int, float, np.number)
"""Type annotation for a numeric type (int, float, np.number)."""

TypeValList = t.Sequence[TypeNumeric]
"""Type annotation for a sequence of numeric type elements."""


def sum_histogram(values: TypeValList, bins: int = 10,
                  normalize: bool = True) -> TypeValList:
    """Returns a list of frequencies of a histogram of given values.

    Args:
        values (:obj:`list` of numerics): a collection of values to con-
            struct the histogram.

        bins (:obj:`int`, optional): number of bins (separations) of the
            histogram.

        normalize(:obj:`bool`, optional): if True, return the relative
            frequencies instead of the abosulte ones.

    Returns:
        list: frequencies of each histogram bin.

    Raises:
        TypeError: if ``values`` contains non-numeric data.
    """

    freqs, _ = np.histogram(values, bins=bins)

    if normalize:
        freqs = freqs / sum(freqs)

    return freqs


def sum_quantiles(values: TypeValList,
                  package: str = "numpy",
                  numpy_interpolation: str = "linear",
                  scipy_alphap: float = 0.4,
                  scipy_betap: float = 0.4) -> TypeValList:
    """Calc. min, first, second and third quartiles, and max from  ``values``.

    Args:
        values (:obj:`sequence` of numerics): values to calculate quartiles.

        package (:obj:`str`, optional): value between ``numpy`` or ``scipy``.
            This argument defines which available package, with a quantile
            calculation implementation, should be used.

        numpy_interpolation (:obj:`str`, optional): if the ``package`` argument
            value is ``numpy``, then this argument is used to define which in-
            terpolation algorithm is used to define the quantiles. The argument
            value must be a value between (``linear``, ``lower``, ``nearest``,
            ``higher``, ``midpoint``).  See ``numpy.percentile`` documentation
            for deeper information about this parameter. If the ``package`` ar-
            gument value isn't ``numpy``, then this argument does not affect
            the output.

        scipy_alphap (:obj:`float`, optional): this argument is used only if
            ``package`` argument value is ``scipy``, which affects this functi-
            on output values directly. Check ``scipy.stats.mstats.mquantiles``
            documentation for deeper information about this and how you should
            configure this

        scipy_betap (:obj:`float`, optional): same as above.

    Returns:
        np.ndarray: values, necessarily in this order, of minimum (0 percenti-
            le), first quartile (25 percentile), median (50 percentile), third
            quartile (75 percentile) and maximum (100 percentile).
    Raises:
        ValueError: if ``package`` value is neither ``numpy`` or ``scipy``, or
            ``numpy_interpolation`` argument has not a valid value.
    """
    valid_packges = ("numpy", "scipy")

    if package not in valid_packges:
        raise ValueError('"package" must be in {} '
                         "(got {}).".format(valid_packges, package))

    if package == "numpy":
        return np.percentile(
            values, (0, 25, 50, 75, 100), interpolation=numpy_interpolation)

    return scipy.stats.mstats.mquantiles(
        values, (0.00, 0.25, 0.50, 0.75, 1.00),
        alphap=scipy_alphap,
        betap=scipy_betap)


def sum_skewness(values: TypeValList, method: int = 3,
                 bias: bool = True) -> float:
    """Calculate the skewness from ``values`` using ``method`` strategy.

    Args:
        values (:obj:`list` or numerics): values from where the skewness
            is estimated.

        method (:obj:`int`, optional): defines the strategy used for es-
            timate data skewness. This argument is used fo compatibility
            with R package ``e1071``. The options must be one of the fol-
            lowing:

            +--------+-----------------------------------------------+
            |Option  | Formula                                       |
            +--------+-----------------------------------------------+
            |1       | Skew_1 = m_3 / m_2**(3/2)                     |
            |        | (default of ``scipy.stats``)                  |
            +--------+-----------------------------------------------+
            |2       | Skew_2 = Skew_1 * sqrt(n(n-1)) / (n-2)        |
            +--------+-----------------------------------------------+
            |3       | Skew_3 = m_3 / s**3 = Skew_1 ((n-1)/n)**(3/2) |
            +--------+-----------------------------------------------+

            Where ``n`` is the number of elements in dataset, ``m_i`` is
            the ith momentum of ``values``, and ``s`` is the standard de-
            viation of ``values``.

            Note that if the selected method is unable to be calculated
            due to division by zero, then the first method will be used
            instead.

        bias (:obj:`bool`, optional): If False, then the calculations are
        corrected for statistical bias.

    Return:
        float: estimated kurtosis from ``values`` using ``method``.

    Raises:
        ValueError: if ``method`` is not 1, 2 nor 3.
    """
    if method not in (1, 2, 3):
        raise ValueError('Invalid method "{}" for '
                         "extracting the skewness".format(method))

    num_vals = len(values)

    if num_vals == 0:
        return np.nan

    skew_val = scipy.stats.skew(values, bias=bias)

    if method == 2 and num_vals != 2:
        skew_val *= (num_vals * (num_vals - 1.0))**0.5 / (num_vals - 2.0)

    elif method == 3:
        skew_val *= ((num_vals - 1.0) / num_vals)**(1.5)

    return skew_val


def sum_kurtosis(values: TypeValList, method: int = 3,
                 bias: bool = True) -> TypeValList:
    """Calculate the kurtosis of ``values`` using ``method`` strategy.

    Args:
        values (:obj:`list` of numeric): values from where the kurtosis is
            estimated.

        method (:obj:`int`, optional): defines the strategy used for esti-
            mate data kurtosis. Used for total compatibility with R package
            ``e1071``. The options must be one of the following:

            +--------+-----------------------------------------------+
            |Option  | Formula                                       |
            +--------+-----------------------------------------------+
            |1       | Kurt_1 = m_4 / m_2**2 - 3                     |
            |        | (default of ``scipy.stats``)                  |
            +--------+-----------------------------------------------+
            |2       | Kurt_2 = ((n+1) * Kurt_1 + 6) * (n-1) / f_2   |
            |        | f_2 = ((n-2)*(n-3))                           |
            +--------+-----------------------------------------------+
            |3       | Kurt_3 = m_4 / s**4 - 3                       |
            |        |        = (Kurt_1+3) * (1 - 1/n)**2 - 3        |
            +--------+-----------------------------------------------+

            Where ``n`` is the number of elements in ``values``, ``s`` is
            the standard deviation of ``values`` and ``m_i`` is the ith
            statistical momentum of ``values``.

            Note that if the selected method is unable to be calculated due
            to division by zero, then the first method is used instead.

        bias (:obj:`bool`): If False, then the calculations are corrected
            for statistical bias.

    Returns:
        float: kurtosis estimated from ``values`` using ``method``.

    Raises:
        ValueError: if ``method`` is not 1, 2 nor 3.
    """
    if method not in (1, 2, 3):
        raise ValueError('Invalid method "{}" for '
                         "extracting the kurtosis".format(method))

    num_vals = len(values)

    if num_vals == 0:
        return np.nan

    kurt_val = scipy.stats.kurtosis(values, bias=bias)

    if method == 2 and num_vals > 3:
        kurt_val = (num_vals + 1.0) * kurt_val + 6
        kurt_val *= (num_vals - 1.0) / ((num_vals - 2.0) * (num_vals - 3.0))

    elif method == 3:
        kurt_val = (kurt_val + 3.0) * (1.0 - 1.0 / num_vals)**2.0 - 3.0

    return kurt_val


SUMMARY_METHODS = collections.OrderedDict((
    ("mean", np.mean),
    ("sd", np.std),
    ("var", np.var),
    ("count", len),
    ("histogram", sum_histogram),
    ("iq_range", scipy.stats.iqr),
    ("kurtosis", sum_kurtosis),
    ("max", max),
    ("median", np.median),
    ("min", min),
    ("quantiles", sum_quantiles),
    ("range", np.ptp),
    ("skewness", sum_skewness),
))
