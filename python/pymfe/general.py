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

    FEATURES = (
        "attr_to_inst",
        "cat_to_num",
        "freq_class",
        "inst_to_attr",
        "nr_attr",
        "nr_bin",
        "nr_cat",
        "nr_class",
        "nr_inst",
        "nr_num",
        "num_to_cat",
    )

    def __init__(self, features: Union[str, Iterable[str]] = "all") -> None:
        """Extracts general metafeatures from datasets.

        Args:
            features: string or list of strings containing the
                metafeatures that should be extracted from fitted
                datasets.
        """
        self.features = features

    @staticmethod
    def ft_nr_inst(X: np.array) -> int:
        """Returns number of instances (rows) in dataset."""
        return X.shape[0]
