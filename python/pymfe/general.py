"""Module dedicated to extraction of General Metafeatures.

Todo:
    - Implement all metafeatures.
    - Improve documentation.

References:
    1. "Towards Reproducible Empirical Research in Meta-Learning",
        Rivolli et al. URL: https://arxiv.org/abs/1808.10406
"""
from typing import Union, Iterable


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

    def inst_num(self, dataframe):
        """Returns number of instances (rows) in DataFrame."""
        return dataframe.shape[0]
