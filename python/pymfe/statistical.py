"""Module dedicated to extraction of Statistical Metafeatures.

Todo:
    * Implement metafeatures.
    * Improve documentation.
"""
import typing as t

import numpy as np
import scipy


class MFEStatistical:
    """To do this documentation."""

    @classmethod
    def ft_can_cor(cls, N: np.ndarray):
        """To do this doc."""
        pass

    @classmethod
    def ft_gravity(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        pass

    @classmethod
    def ft_cor(cls, N: np.ndarray) -> t.Union[float, np.ndarray]:
        """Absolute value of correlation between distinct column pairs."""
        corr_mat = np.corrcoef(N, rowvar=False)

        res_num_rows, _ = corr_mat.shape

        inf_triang_vals = corr_mat[np.tril_indices(res_num_rows, k=-1)]

        return abs(inf_triang_vals)

    @classmethod
    def ft_cov(cls, N: np.ndarray) -> np.ndarray:
        """Absolute value of covariance between distinct column pairs."""
        cov_mat = np.cov(N, rowvar=False)

        res_num_rows, _ = cov_mat.shape

        inf_triang_vals = cov_mat[np.tril_indices(res_num_rows, k=-1)]

        return abs(inf_triang_vals)

    @classmethod
    def ft_nr_disc(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        pass

    @classmethod
    def ft_eigenvalues(cls, N: np.ndarray) -> t.Union[float, np.ndarray]:
        """Returns eigenvalues of covariance matrix."""
        cov_mat = np.cov(N, rowvar=False)

        try:
            eigvals = np.linalg.eigvals(cov_mat)

        except np.linalg.LinAlgError:
            return np.nan

        return eigvals

    @classmethod
    def ft_g_mean(cls, N: np.ndarray) -> np.ndarray:
        """Geometric mean of each column."""
        return scipy.stats.mstats.gmean(N, axis=0)

    @classmethod
    def ft_h_mean(cls, N: np.ndarray) -> np.ndarray:
        """Harmonic mean of each column."""
        return scipy.stats.mstats.hmean(N, axis=0)

    @classmethod
    def ft_iq_range(cls, N: np.ndarray) -> np.ndarray:
        """Compute Interquartile Range (IQR) of each column."""
        return scipy.stats.iqr(N, axis=0)

    @classmethod
    def ft_kurtosis(cls, N: np.ndarray, bias: bool = False) -> np.ndarray:
        """To do this doc."""
        return scipy.stats.kurtosis(N, axis=0, bias=bias)

    @classmethod
    def ft_mad(cls, N: np.ndarray, factor: float = 1.4826) -> np.ndarray:
        """Computes Median Absolute Deviation (MAD) adjusted by a factor.

        The default ``factor`` is 1.4826 due to fact that it is an appro-
        ximated result of MAD of a normally distributed data, so it make
        this method result comparable with this sort of data.
         """
        median_dev = abs(N - np.median(N, axis=0))
        return np.median(median_dev, axis=0) * factor

    @classmethod
    def ft_max(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        return N.max(axis=0)

    @classmethod
    def ft_mean(cls, N: np.ndarray) -> np.ndarray:
        """Returns the mean value of each data column."""
        return N.mean(axis=0)

    @classmethod
    def ft_median(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        return np.median(N, axis=0)

    @classmethod
    def ft_min(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        return N.min(axis=0)

    @classmethod
    def ft_nr_cor_attr(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        pass

    @classmethod
    def ft_nr_norm(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        pass

    @classmethod
    def ft_nr_outliers(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        pass

    @classmethod
    def ft_range(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        return np.ptp(N, axis=0)

    @classmethod
    def ft_sd(cls, N: np.ndarray, ddof: float = 1) -> np.ndarray:
        """To do this doc."""
        return N.std(axis=0, ddof=ddof)

    @classmethod
    def ft_sd_ratio(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        pass

    @classmethod
    def ft_skewness(cls, N: np.ndarray, bias: bool = False) -> np.ndarray:
        """To do this doc."""
        return scipy.stats.skew(N, axis=0, bias=bias)

    @classmethod
    def ft_sparcity(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        pass

    @classmethod
    def ft_t_mean(cls, N: np.ndarray,
                  pcut: float = 0.2) -> t.Union[float, np.ndarray]:
        """To do this doc."""
        return scipy.stats.trim_mean(N, proportiontocut=pcut)

    @classmethod
    def ft_var(cls, N: np.ndarray, ddof: float = 1) -> np.ndarray:
        """To do this doc."""
        return N.var(axis=0, ddof=ddof)

    @classmethod
    def ft_w_lambda(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        pass


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()

    res = MFEStatistical.ft_t_mean(iris.data)

    print(res)
