"""Module dedicated to extraction of General Metafeatures.

Notes:
    More description about the metafeatures implemented here,
    check out [1].

References:
    [1] "Towards Reproducible Empirical Research in Meta-Learning",
        Rivolli et al. URL: https://arxiv.org/abs/1808.10406

Todo:
    * Implement all metafeatures.
    * Improve documentation.

"""
from typing import Union, Iterable
import numpy as np


class MFEGeneral:
    """General-type Metafeature extractor."""

    def __init__(self, features: Union[str, Iterable[str]] = "all") -> None:
        """Extracts general metafeatures from datasets.

        Args:
            features: string or list of strings containing the
                metafeatures that should be extracted from fitted
                datasets.
        """
        self.features = features

    @classmethod
    def ft_nr_inst(cls, X: np.array) -> int:
        """Returns number of instances (rows) in dataset."""
        return X.shape[0]