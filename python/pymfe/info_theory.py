"""Module dedicated to extraction of Information Theory Metafeatures.

Todo:
    * Implement metafeatures.
    * Improve documentation.
"""
from typing import Union, List

import numpy as np
import scipy


class MFEInfoTheory:
    """To do this documentation."""

    @classmethod
    def _entropy(cls, values: Union[np.ndarray, List]) -> float:
        """Calculate entropy within array 'values'."""
        _, counts = np.unique(values, return_counts=True)
        return scipy.stats.entropy(counts, base=2)

    @classmethod
    def ft_attr_ent(cls, X: Union[np.ndarray, List]) -> np.ndarray:
        """Calculates entropy for each attribute."""
        return np.apply_along_axis(
            func1d=MFEInfoTheory._entropy,
            axis=0,
            arr=X)

    @classmethod
    def ft_class_ent(cls, y: Union[np.ndarray, List]) -> np.ndarray:
        """Calculates class entropy.

        If data has multiclassed instances, each column is interpreted
        as a class.
        """

        if isinstance(y, np.ndarray):
            y = np.array(y)

        if len(y.shape) > 1:
            return MFEInfoTheory.ft_attr_ent(y)

        return MFEInfoTheory._entropy(y)
