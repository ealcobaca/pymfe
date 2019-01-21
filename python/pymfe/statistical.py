"""Module dedicated to extraction of Statistical Metafeatures.

Todo:
    * Implement metafeatures.
    * Improve documentation.
"""
import typing as t

import numpy as np
import scipy.stats


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
    def ft_kurtosis(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        pass

    @classmethod
    def ft_mad(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        pass

    @classmethod
    def ft_max(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        pass

    @classmethod
    def ft_mean(cls, N: np.ndarray) -> np.ndarray:
        """Returns the mean value of each data column."""
        return N.mean(axis=0)

    @classmethod
    def ft_median(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        pass

    @classmethod
    def ft_min(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        pass

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
        pass

    @classmethod
    def ft_sd(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        pass

    @classmethod
    def ft_sd_ratio(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        pass

    @classmethod
    def ft_skewness(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        pass

    @classmethod
    def ft_sparcity(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        pass

    @classmethod
    def ft_t_mean(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        pass

    @classmethod
    def ft_var(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        pass

    @classmethod
    def ft_w_lambda(cls, N: np.ndarray) -> np.ndarray:
        """To do this doc."""
        pass


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()

    res = MFEStatistical.ft_iq_range(iris.data)

    print(res)
