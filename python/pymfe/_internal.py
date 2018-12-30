"""Provides useful functions for MFE package.

Attributes:
    VALID_GROUPS (:obj:`tuple` of :obj:`str`): Supported type of
        metafeatures of pymfe.
    VALID_SUMMARY (:obj:`tuple` of :obj:`str`): Supported summary
        functions to combine metafeature values.
"""
import collections

import numpy as np


VALID_GROUPS = (
    "landmarking",
    "general",
    "statistical",
    "model-based",
    "info-theory",
)

VALID_SUMMARY = (
    "mean",
    "sd",
)


def _check_value_in_group(value, group, wildcard="all"):
    """Checks if a given value (or a collection) are in a group.

    Args:
        value (:obj:`Iterable` of :obj:`str` or :obj:`str`): value(s)
            to be checked if are in the given group of strings.
        group (:obj:`Iterable` of :obj:`str`): a group of strings.

    Returns:
        Tuples: in_group and not_in_group containing, respectivelly,
        values that are in the given group and those that are not.
        If no value is in either group, then this group will be None.

    Raises:
        TypeError: if 'value' is not a Iterable type or some of its
            elements are not a 'str' type.
    """

    if not isinstance(value, collections.Iterable):
        raise TypeError("Parameter type is not consistent.")

    in_group, not_in_group = None, None

    if isinstance(value, str):
        value = value.lower()
        if value == wildcard:
            in_group = group

        elif value in group:
            in_group = (value,)

        else:
            not_in_group = (value,)

    elif isinstance(value, collections.Iterable):
        value = set(map(str.lower, value))

        if wildcard in value:
            in_group = group

        else:
            in_group = value.intersection(group)
            not_in_group = value.difference(group)

    return in_group, not_in_group


def process_groups(groups):
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


def process_summary(summary):
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


def process_features(features):
    """Check if 'features' argument from MFE.__init__ is correct.

    Needs to be implemented.
    """
    return features


def check_data(X, y):
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
