"""Main module for extracting metafeatures from datasets.

Todo:
    * Improve documentation.
    * Implement MFE class.
"""
from typing import Union, Iterable, Sequence, Optional, TypeVar
import collections

import numpy as np

import _internal
# import general
# import statistical
# import info_theory
# import landmarking
# import model_based

TYPE_NUMERIC = TypeVar(int, float)
TYPE_MFECLASSES = TypeVar(_internal.VALID_MFECLASSES)


class MFE:
    """Core class for metafeature extraction."""

    def __init__(self,
                 groups: Union[str, Iterable[str]] = "all",
                 features: Union[str, Iterable[str]] = "all",
                 summary: Union[str, Iterable[str]] = "all") -> None:
        """To do this documentation."""

        self.groups = _internal.process_groups(groups)
        self.features = _internal.process_features(features)
        self.summary = _internal.process_summary(summary)
        self.X = None
        self.y = None
        self.cv_splits = None

    def fit(self,
            X: Sequence,
            y: Sequence,
            splits: Optional[Iterable] = None) -> None:
        """Fits dataset into the MFE model.

        Args:
            X: predictive attributes of the dataset.
            y: target attributes of the dataset.
            splits: iterable which contains K-Fold Cross Validation index
                splits to use mainly in landmarking metafeatures. If not
                given, each metafeature will be extracted a single time.

        Raises:
            ValueError: if number of rows of X and y does not match.
            TypeError: if X or y (or both) is neither a :obj:`list` or
                a :obj:`np.array` object.
        """

        self.X, self.y = _internal.check_data(X, y)

        if not isinstance(splits, collections.Iterable):
            raise TypeError('"splits" argument must be a iterable.')

        self.cv_splits = splits

    def extract(self):
        """Extracts metafeatures from fitted dataset."""
        if self.X is None or self.y is None:
            raise TypeError('Fitted data not found. Call '
                            '"fit" method before "extract".')

        if (not isinstance(self.X, np.array) or
                not isinstance(self.y, np.array)):
            self.X, self.y = _internal.check_data(self.X, self.y)

        # To do.

    @staticmethod
    def _call_feature(feature: str,
                      group_class: TYPE_MFECLASSES,
                      **kwargs) -> Sequence[TYPE_NUMERIC]:
        """Calls a specific feature-related method from class 'group_class'.

        Args:
            feature: feature name. Check out 'FEATURES' attribute of each
                feature extractor class for possible valid values.
            group_class: should be a feature extractor class. Current valid
                values are listed below:
                    - MFEGeneral: General/Basic features class.
                    - MFEInfoTheory: Information theory features class.
                    - MFEStatistical: Statistical features class.
                    - MFELandmarking: Landmarking features class.
                    - MFEModelBased: Model-based features class.
            **kwargs: arguments for the called feature method.

        Returns:
            Invoked method return value.

        Raises:
            AttributeError: if specified method does not exists in the
                given class or given class is not valid.
            Any Exception raised by the method invoked can also be raised
                by this method.
        """
        return getattr(group_class, "ft_{0}".format(feature))(**kwargs)
