"""Provides useful functions for MFE package.

Attributes:
    VALID_GROUPS (:obj:`tuple` of :obj:`str`): Supported type of
        metafeatures of pymfe.
    VALID_SUMMARY (:obj:`tuple` of :obj:`str`): Supported summary
        functions to combine metafeature values.
    VALID_MFECLASSES (:obj:`tuple`): Metafeature extractors classes.

Todo:
    * Implement "check_features" function
"""
from typing import Union, Tuple, Iterable, Generic, List, Dict
import inspect
import collections

import numpy as np

import general
import statistical
import info_theory
import landmarking
import model_based

VALID_GROUPS = (
    "landmarking",
    "general",
    "statistical",
    "model-based",
    "info-theory",
)  # type: Tuple[str, ...]

VALID_SUMMARY = (
    "mean",
    "sd",
    "count",
    "histogram",
    "iq_range",
    "kurtosis",
    "max",
    "median",
    "min",
    "quartiles",
    "range",
    "skewness",
)  # type: Tuple[str, ...]

VALID_MFECLASSES = (
    landmarking.MFELandmarking,
    general.MFEGeneral,
    statistical.MFEStatistical,
    model_based.MFEModelBased,
    info_theory.MFEInfoTheory,
)  # type: Tuple


def _check_value_in_group(
        value: Union[str, Iterable[str]],
        group: Iterable[str],
        wildcard: str = "all") -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Checks if a value is in a set or a set of values is a subset of a set.

    Args:
        value: value(s) to be checked if are in the given group of strings.
        group: a group of strings.

    Returns:
        A pair of tuples containing, respectivelly, values that are in
        the given group and those that are not. If no value is in either
        group, then this group will be None.

    Raises:
        TypeError: if 'value' is not a Iterable type or some of its
            elements are not a 'str' type.
    """

    if not isinstance(value, collections.Iterable):
        raise TypeError("Parameter type is not "
                        "consistent ({0}).".format(type(value)))

    in_group = tuple()  # type: Tuple[str, ...]
    not_in_group = tuple()  # type: Tuple[str, ...]

    if isinstance(value, str):
        value = value.lower()
        if value == wildcard:
            in_group = tuple(group)

        elif value in group:
            in_group = (value, )

        else:
            not_in_group = (value, )

    else:
        value_set = set(map(str.lower, value))

        if wildcard in value_set:
            in_group = tuple(group)

        else:
            in_group = tuple(value_set.intersection(group))
            not_in_group = tuple(value_set.difference(group))

    return in_group, not_in_group


def process_groups(groups: Union[Iterable[str], str]) -> Tuple[str, ...]:
    """Check if 'groups' argument from MFE.__init__ is correct.

    Args:
        groups (:obj:`str` or :obj:`Iterable` of :obj:`str`): a single
            string or a iterable with group identifiers to be processed.
            It must assume or contain the following values:
                - 'landmarking': Landmarking metafeatures.
                - 'general': General and Simple metafeatures.
                - 'statistical': Statistical metafeatures.
                - 'model-based': Metafeatures from machine learning models.
                - 'info-theory': Information Theory metafeatures.

    Returns:
        A tuple containing all valid group lower-cased identifiers.

    Raises:
        TypeError: if 'groups' is neither a string 'all' nor a Iterable
            containing valid group identifiers as strings.
        ValueError: if 'groups' is None or is a empty Iterable or
            if a unknown group identifier is given.
    """
    if not groups:
        raise ValueError('"Groups" can not be None nor empty.')

    in_group, not_in_group = _check_value_in_group(groups, VALID_GROUPS)

    if not_in_group:
        raise ValueError("Unknown groups: {0}".format(not_in_group))

    return in_group


def process_summary(summary: Union[str, Iterable[str]]) -> Tuple[str, ...]:
    """Check if 'summary' argument from MFE.__init__ is correct.

    Args:
        summary (:obj:`Iterable` of :obj:`str` or a :obj:`str`): a
            summary function or a list of these, which are used to
            combine different calculations of the same metafeature.
            Check out reference [1] for more information.
            The values must be one of the following:
                - 'mean': Average of the values.
                - 'sd': Standard deviation of the values.
                - 'count': Computes the cardinality of the measure.
                    Suitable for variable cardinality.
                - 'histogram': Describes the distribution of the mea-
                    sure values. Suitable for high cardinality.
                - 'iq_range': Computes the interquartile range of the
                    measure values.
                - 'kurtosis': Describes the shape of the measures values
                    distribution.
                - 'max': Resilts in the maximum vlaues of the measure.
                - 'median': Results in the central value of the measure.
                - 'min': Results in the minimum value of the measure.
                - 'quartiles': Results in the minimum, first quartile,
                    median, third quartile and maximum of the measure
                    values.
                - 'range': Computes the ranfe of the measure values.
                - 'skewness': Describes the shaoe of the measure values
                    distribution in terms of symmetry.

    Raises:
        TypeError: if 'summary' is neither a string 'all' nor a Iterable
            containing valid group identifiers as strings.
        ValueError: if 'summary' is None or is a empty Iterable or
            if a unknown group identifier is given.

    Returns:
        A tuple containing all valid lower-cased summary functions.

    References:
        [1] "Towards Reproducible Empirical Research in Meta-Learning",
                Rivolli et al. URL: https://arxiv.org/abs/1808.10406
    """
    if not summary:
        raise ValueError('"Summary" can not be None nor empty.')

    in_group, not_in_group = _check_value_in_group(summary, VALID_SUMMARY)

    if not_in_group:
        raise ValueError("Unknown groups: {0}".format(not_in_group))

    return in_group


def process_features(features: Union[str, Iterable[str]]) -> Tuple[str, ...]:
    """Check if 'features' argument from MFE.__init__ is correct.

    Needs to be properly implemented.
    """
    return tuple(features)


def check_data(X: Union[np.array, list],
               y: Union[np.array, list]) -> Tuple[np.array, np.array]:
    """Checks received data type and shape.

    Args:
        Check "mfe.fit" method for more information.

    Raises:
        Check "mfe.fit" method for more information.

    Returns:
        X and y both casted to a numpy.array.
    """
    if not isinstance(X, (np.array, list)):
        raise TypeError('"X" is neither "list" nor "np.array".')

    if not isinstance(y, (np.array, list)):
        raise TypeError('"y" is neither "list" nor "np.array".')

    if not isinstance(X, np.array):
        X = np.array(X)

    if not isinstance(y, np.array):
        y = np.array(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError('"X" and "y" shapes (number of rows) do not match.')

    return X, y


def get_feature_methods(
        class_address: Generic) -> List[Tuple[str, Generic]]:
    """Get feature-extraction related methods from a given MFE Class.

    Methods related with feature extraction is assumed to be prefixed
    with "ft_".

    Args:
        class_address: Class address from which the feature methods
            should be extracted.

    Returns:
        A list of tuples in the form ('method_name', 'method_address')
        which contains all methods associated with feature extraction
        (prefixed with "ft_").
    """
    feature_method_list = inspect.getmembers(class_address,
                                             predicate=inspect.ismethod)

    # It is assumed that all feature-extraction related methods
    # name are all prefixed with "ft_".
    feature_method_list = [
        ft_method for ft_method in feature_method_list
        if ft_method[0].startswith("ft_")
    ]

    return feature_method_list


def get_all_feature_methods() -> Dict[str, List[Tuple[str, Generic]]]:
    """Get all feature-extraction related methods from all Classes.

    Returns:
        Dict in the form {'group_name': [('method_name', 'method_address')]},
        i.e. the keys are the names of feature groups (e.g. 'general' or
        'landmarking') and values are lists of tuples which first entry are
        feature-extraction related method names and the second entry are its
        correspondent address.

    Example:
        {
            'general': [
                ('ft_nr_num', <method_address>),
                ('ft_nr_inst', <method_address>),
                ...
            ],

            'statistical': [
                ('ft_mean', <method_address>),
                ('ft_max', <method_address>),
                ...
            ],

            ...
        }
    """
    feature_method_dict = {
        ft_type_id: get_feature_methods(mfe_class)
        for ft_type_id, mfe_class in zip(VALID_GROUPS, VALID_MFECLASSES)
    }  # type: Dict[str, List[Tuple[str, Generic]]]

    return feature_method_dict
