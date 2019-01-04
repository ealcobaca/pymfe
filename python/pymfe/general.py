"""Module dedicated to extraction of General Metafeatures.

Notes:
    More description about the metafeatures implemented here,
    check out `Rivolli et al.`_.

References:
    .. _Rivolli et al.:
        "Towards Reproducible Empirical Research in Meta-Learning",
        Rivolli et al. URL: https://arxiv.org/abs/1808.10406

Todo:
    * Implement all metafeatures.
    * Improve documentation.

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
    def ft_nr_inst(cls, X: np.array) -> int:
        """Returns number of instances (rows) in dataset."""
        return X.shape[0]
