"""Module dedicated to extraction of Information Theory Metafeatures.

Todo:
    * Implement metafeatures.
    * Improve documentation.
"""
import typing as t

import numpy as np
import scipy


class MFEInfoTheory:
    """To do this documentation."""

    @classmethod
    def _entropy(cls, values: t.Union[np.ndarray, t.List]) -> float:
        """Calculate entropy within array 'values'."""
        _, counts = np.unique(values, return_counts=True)
        return scipy.stats.entropy(counts, base=2)

    @classmethod
    def ft_attr_ent(cls, C: np.ndarray) -> t.Union[np.ndarray, np.float]:
        """Calculates entropy for each attribute."""
        try:
            return np.apply_along_axis(
                func1d=MFEInfoTheory._entropy,
                axis=0,
                arr=C)

        except ValueError:
            return np.nan

    @classmethod
    def ft_class_ent(cls, y: np.ndarray) -> t.Union[np.ndarray, np.float]:
        """Calculates class entropy.

        If data has multiclassed instances, each column is interpreted
        as a class.
        """
        if len(y.shape) > 1:
            return MFEInfoTheory.ft_attr_ent(y)

        return MFEInfoTheory._entropy(y)
