# -*- coding: utf-8 -*-
"""Main module for extracting metafeatures from datasets.

Todo:
    * Improve documentation.
    * Implement MFE class.
"""
from typing import Union, Iterable, Sequence, \
    Optional, Any, Dict, Callable
from typing import Tuple, List, Generator  # noqa: F401
import collections
import warnings

import numpy as np

import _internal


class MFE:
    """Core class for metafeature extraction."""

    def __init__(self,
                 groups: Union[str, Iterable[str]] = "all",
                 features: Union[str, Iterable[str]] = "all",
                 summary: Union[str, Iterable[str]] = ("mean", "sd")) -> None:
        """To do this documentation."""

        self.groups = _internal.process_groups(groups)
        self.features = _internal.process_features(features)  \
            # type: Sequence[Tuple[str, Callable, Sequence]]


        self.summary = _internal.process_summary(summary)  \
            # type: Sequence[Tuple[str, Callable]]

        self.X = None  # type: Optional[np.array]
        self.y = None  # type: Optional[np.array]
        self.cv_splits = None  # type: Optional[Iterable[int]]

    @classmethod
    def _summarize(cls,
                   features: Union[np.ndarray, Sequence],
                   sum_callable: Callable,
                   remove_nan: bool = True) -> _internal.TypeNumeric:
        """Returns summarized features, if needed, or feature otherwise.

        Args:
            features: Sequence containing values to summarize.
            sum_callable: Callable of the method which implements the desired
                summary function.
            remove_nan: check and remove all elements in 'features' which are
                not numeric ('int' or 'float' types). Note that :obj:`np.inf`
                is still considered numeric.

        Returns:
            Float value of summarized feature values if possible. May return
            :obj:`np.nan` if summary function call invokes TypeError.

        Raises:
            AttributeError: if 'sum_callable' is invalid.
        """
        if isinstance(features, (np.ndarray, collections.Sequence)):

            processed_feat = np.array(features)

            if remove_nan:
                numeric_vals = list(map(_internal.isnumeric, features))
                processed_feat = processed_feat[numeric_vals]
                processed_feat = processed_feat.astype(np.float32)

            try:
                metafeature = sum_callable(processed_feat)

            except TypeError:
                metafeature = np.nan

        else:
            # Equivalent to identity summary function: f(x) = x
            if _internal.isnumeric(features):
                metafeature = features

            else:
                metafeature = np.nan

        return metafeature

    def fit(self,
            X: Sequence,
            y: Sequence,
            splits: Optional[Iterable[int]] = None) -> None:
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

        if (splits is not None
                and not isinstance(splits, collections.Iterable)):
            raise TypeError('"splits" argument must be a iterable.')

        self.cv_splits = splits

    def _build_ft_mtd_args(self,
                           ft_mtd_name: str,
                           ft_mtd_args: Iterable[str],
                           user_custom_args: Optional[Dict[str, Any]],
                           suppress_warnings=False) -> Dict[str, Any]:
        """Build a 'kwargs' dict for a feature-extraction callable.

        Args:
            ft_mtd_args: Iterable containing the name of all arguments of
                the feature-extraction callable.
            user_custom_args: Dict in the form {'arg': value} given by
                user to customize the given feature-extraction callable.
            suppress_warnings: do not show any warnings about unknown
                parameters.

        Returns:
            A Dict which is a ready-to-use kwargs for the correspondent
            callable. Note that is expected that the feature-extraction
            callable only has at maximum "X" and "y" user-independent
            obligatory arguments.
        """
        if user_custom_args:
            callable_args = {
                custom_arg: user_custom_args[custom_arg]
                for custom_arg in user_custom_args if custom_arg in ft_mtd_args
            }

        else:
            callable_args = dict()

        if not suppress_warnings:
            unknown_arg_set = (unknown_arg
                               for unknown_arg in callable_args.keys()
                               if unknown_arg not in ft_mtd_args
                               )  # type: Generator[str, None, None]

            for unknown_arg in unknown_arg_set:
                warnings.warn(
                    "Unknown argument {0} for method {1}.".format(
                        unknown_arg, ft_mtd_name), UserWarning)

        if "X" in ft_mtd_args:
            callable_args["X"] = self.X

        if "y" in ft_mtd_args:
            callable_args["y"] = self.y

        return callable_args

    @staticmethod
    def _get_feat_value(method_name: str,
                        method_args: Dict[str, Any],
                        method_callable: Callable,
                        suppress_warnings: bool = False
                        ) -> Union[_internal.TypeNumeric, np.ndarray]:
        """Extract feat. from 'method_callable' with 'method_args' as args."""

        try:
            features = method_callable(**method_args)

        except TypeError as type_e:
            if not suppress_warnings:
                warnings.warn(
                    "Error extracting {0}: \n{1}.\nWill set it "
                    "as 'np.nan' for all summary functions.".format(
                        method_name, repr(type_e)), RuntimeWarning)

            features = np.nan

        return features

    def extract(self, suppress_warnings: bool = False,
                **kwargs) -> Tuple[List[str], List[float]]:
        """Extracts metafeatures from previously fitted dataset.

        Args:
            suppress_warnings: do not show warnings about unknown user
                custom parameters.

        Returns:
            List containing all metafeatures summarized by all summary
            functions loaded in the model.

        Raises:
            TypeError: if calling 'extract' method before 'fit' method.
        """
        if self.X is None or self.y is None:
            raise TypeError("Fitted data not found. Call "
                            '"fit" method before "extract".')

        if (not isinstance(self.X, np.ndarray)
                or not isinstance(self.y, np.ndarray)):
            self.X, self.y = _internal.check_data(self.X, self.y)

        metafeat_vals = []  # type: List[Union[int, float]]
        metafeat_names = []  # type: List[str]
        for ft_mtd_name, ft_mtd_callable, ft_mtd_args in self.features:

            mtd_args_pack = self._build_ft_mtd_args(
                ft_mtd_name, ft_mtd_args, kwargs.get(ft_mtd_name),
                suppress_warnings)  # type: Dict[str, Any]

            features = MFE._get_feat_value(
                ft_mtd_name,
                mtd_args_pack,
                ft_mtd_callable,
                suppress_warnings)  \
                # type: Union[np.ndarray, Sequence, float, int]

            if isinstance(features, (np.ndarray, collections.Sequence)):
                for sum_mtd_name, sum_mtd_callable in self.summary:
                    summarized_val = MFE._summarize(features, sum_mtd_callable)
                    metafeat_vals.append(summarized_val)
                    metafeat_names.append("{0}.{1}".format(
                        ft_mtd_name, sum_mtd_name))

            else:
                metafeat_vals.append(features)
                metafeat_names.append(ft_mtd_name)

        return metafeat_names, metafeat_vals

    @staticmethod
    def _call_feature(feature: str, group_class,
                      **kwargs) -> Sequence[_internal.TypeNumeric]:
        """Calls a specific feature-related method from class 'group_class'.

        Args:
            feature: feature name. Check out 'FEATURES' attribute of each
                feature extractor class for possible valid values.
            group_class: should be a feature extractor class. Current valid
                values are listed below:
                    1. MFEGeneral: General/Basic features class.
                    2. MFEInfoTheory: Information theory features class.
                    3. MFEStatistical: Statistical features class.
                    4. MFELandmarking: Landmarking features class.
                    5. MFEModelBased: Model-based features class.
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


X = np.array([
    [1, 2, 3, 4],
    [0, 0, 1, 1],
    [2, 2, 1, 1],
    [1, 1, 1, 1],
])

y = np.array([1, 1, 0, 0])

m = MFE()
m.fit(X=X, y=y)
names, vals = m.extract()
for n, v in zip(names, vals):
    print(n, v)
