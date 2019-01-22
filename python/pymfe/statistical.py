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
    def ft_can_cor(cls, N: np.ndarray, y: np.ndarray) -> np.ndarray:
        """To do this doc."""

    @classmethod
    def ft_gravity(cls, N: np.ndarray, y: np.ndarray) -> np.ndarray:
        """To do this doc."""
        classes, freqs = np.unique(y, return_counts=True)

        class_freq_most, _ = max(zip(classes, freqs), key=lambda x: x[1])

        class_freq_most_ind = np.where(class_freq_most == classes)[0]

        classes = np.delete(classes, class_freq_most_ind)
        freqs = np.delete(freqs, class_freq_most_ind)

        class_freq_least, _ = min(zip(classes, freqs), key=lambda x: x[1])

        center_freq_class_most = N[y == class_freq_most, :].mean(axis=0)
        center_freq_class_least = N[y == class_freq_least, :].mean(axis=0)

        return np.linalg.norm(
            center_freq_class_most - center_freq_class_least, ord=2)

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
    def ft_nr_cor_attr(cls,
                       N: np.ndarray,
                       threshold: float = 0.5,
                       normalize: bool = True) -> np.ndarray:
        """Number of attribute pairs with corr. equal or greater than a threshold.

        Args:
            threshold (:obj:`float`, optional): value of threshold, where
                correlation is assumed to be strong if its absolute value
                is equal or greater than it.

            normalize (:obj:`bool`, optional): if True, the result will be
                normalized by a factor of 2 / (d * (d - 1)), whered = number
                of attributes (columns) in N.
        """
        abs_corr_vals = MFEStatistical.ft_cor(N)

        _, num_attr = N.shape

        norm_factor = 1.0

        if normalize:
            norm_factor = 2.0 / (num_attr * (num_attr - 1.0))

        return sum(abs_corr_vals >= threshold) * norm_factor

    @classmethod
    def ft_nr_norm(cls, N: np.ndarray, threshold: float = 0.1) -> int:
        """To do this doc."""
        nr_norm = 0

        for attr in N.T:
            _, p_value = scipy.stats.shapiro(attr)
            nr_norm += p_value < threshold

        return nr_norm

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
    def ft_sparsity(cls, N: np.ndarray, normalize: bool = True) -> np.ndarray:
        """To do this doc."""

        ans = np.array([attr.size / np.unique(attr).size for attr in N.T])

        norm_factor = 1.0
        if normalize:
            norm_factor = 1.0 / (N.shape[0] - 1.0)

        return (ans - 1.0) * norm_factor

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

    res = MFEStatistical.ft_sd_ratio(iris.data)

    print(res)
