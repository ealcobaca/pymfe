"""Provides useful functions for MFE package.

Attributes:
    VALID_GROUPS (:obj:`tuple` of :obj:`str`): Supported type of
        metafeatures of pymfe.
    VALID_SUMMARY (:obj:`tuple` of :obj:`str`): Supported summary
        functions to combine metafeature values.
    VALID_MFECLASSES (:obj:`tuple): Metafeature extractors classes.

Todo:
    * Implement "check_features" function
"""
from typing import Union, Tuple, Iterable
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
)  # type: Tuple[str, ...]

VALID_MFECLASSES = (
    general.MFEGeneral,
    statistical.MFEStatistical,
    info_theory.MFEInfoTheory,
    landmarking.MFELandmarking,
    model_based.MFEModelBased,
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
            The values must be one of the following:
                - "mean": average of the values.
                - "sd": standard deviation of the values.
                - more to be implemented (...)

    Raises:
        TypeError: if 'summary' is neither a string 'all' nor a Iterable
            containing valid group identifiers as strings.
        ValueError: if 'summary' is None or is a empty Iterable or
            if a unknown group identifier is given.

    Returns:
        A tuple containing all valid lower-cased summary functions.
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
