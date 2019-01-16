"""Module dedicated to extraction of General Metafeatures.

Notes:
    More description about the metafeatures implemented here,
    check out `Rivolli et al.`_.

References:
    .. _Rivolli et al.:
        "Towards Reproducible Empirical Research in Meta-Learning",
        Rivolli et al. URL: https://arxiv.org/abs/1808.10406
"""
import typing as t
import numpy as np


class MFEGeneral:
    """General-type Metafeature extractor."""

    def __init__(self,
                 features: t.Union[str, t.Iterable[str]] = "all") -> None:
        """Extracts general metafeatures from datasets.

        Args:
            features: string or list of strings containing the
                metafeatures that should be extracted from fitted
                datasets.
        """
        self.features = features

    @classmethod
    def ft_attr_to_inst(cls, X: np.ndarray) -> int:
        """Returns ration between number of attributes and instances.

        It is effectively the inverse of value given by ``ft_inst_to_attr``.
        """
        return X.shape[1] / X.shape[0]

    @classmethod
    def ft_cat_to_num(cls, C: np.ndarray,
                      N: np.ndarray) -> t.Union[int, np.float]:
        """Returns ratio between number of categoric and numeric features.

        If number of numeric features is zero, :obj:`np.nan` is returned
        instead.

        Effectively the inverse of value given by ``ft_num_to_cat``.
        """
        if N.shape[1] == 0:
            return np.nan

        return C.shape[1] / N.shape[1]

    @classmethod
    def ft_freq_class(cls, y: np.ndarray) -> np.ndarray:
        """Returns an array of frequency of each distinct class."""
        _, freq = np.unique(y, return_counts=True)
        return freq

    @classmethod
    def ft_inst_to_attr(cls, X: np.ndarray) -> int:
        """Returns ratio between number of instances and attributes.

        It is effectively the inverse of value given by ``ft_attr_to_inst``.
        """
        return X.shape[0] / X.shape[1]

    @classmethod
    def ft_nr_attr(cls, X: np.ndarray) -> int:
        """Returns number of total attributes."""
        return X.shape[1]

    @classmethod
    def ft_nr_bin(cls, X: np.ndarray) -> int:
        """Returns number of binary attributes."""
        bin_cols = np.apply_along_axis(
            func1d=lambda col: np.unique(col).size == 2, axis=0, arr=X)

        return sum(bin_cols)

    @classmethod
    def ft_nr_cat(cls, C: np.ndarray) -> int:
        """Returns number of categorical attributes."""
        return C.shape[1]

    @classmethod
    def ft_nr_class(cls, y: np.ndarray) -> int:
        """Returns number of distinct classes."""
        return np.unique(y).size

    @classmethod
    def ft_nr_inst(cls, X: np.ndarray) -> int:
        """Returns number of instances (rows) in dataset."""
        return X.shape[0]

    @classmethod
    def ft_nr_num(cls, N: np.ndarray) -> int:
        """Returns number of numeric features."""
        return N.shape[1]

    @classmethod
    def ft_num_to_cat(cls, C: np.ndarray,
                      N: np.ndarray) -> t.Union[int, np.float]:
        """Returns ratio between number of numeric and categoric features.

        If number of categoric features is zero, :obj:`np.nan` is returned
        instead.

        Effectively the inverse of value given by ``ft_cat_to_num``.
        """
        if C.shape[1] == 0:
            return np.nan

        return N.shape[1] / C.shape[1]
