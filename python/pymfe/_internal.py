"""Provides useful functions for MFE package.

Attributes:
    VALID_GROUPS (:obj:`tuple` of :obj:`str`): Supported type of
        metafeatures of pymfe.
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
        TypeError: if `groups` is not a string "all", a Iterable
            containing valid group identifiers as strings, is None or
            is a empty Iterable.
        ValueError: if a unknown group identifier is given.
    """
    unknown_groups = None

    if groups is None or not groups:
        raise AttributeError('"Groups" can not be None nor empty.')

    if isinstance(groups, str):
        groups = groups.lower()
        if groups == "all":
            return VALID_GROUPS

        if groups in VALID_GROUPS:
            return (groups,)

        unknown_groups = {groups}

    elif isinstance(groups, collections.Iterable):
        groups = set(map(str.lower, groups))

        if "all" in groups:
            return VALID_GROUPS

        if groups.issubset(VALID_GROUPS):
            return tuple(groups)

        unknown_groups = groups.difference(VALID_GROUPS)

    if unknown_groups is not None:
        raise ValueError("Unknown groups: {0}".format(unknown_groups))

    raise TypeError('"Groups" parameter type is not consistent.')


def process_summary(summary):
    """Check if 'summary' argument from MFE.__init__ is correct."""
    return summary


def process_features(features):
    """Check if 'features' argument from MFE.__init__ is correct."""
    return features


def check_data(X, y):
    """Checks received data type and shape.

    Args:
        Check "mfe.fit" method for more information.

    Raises:
        Check "mfe.fit" method for more information.

    Returns:
        X and y both casted to a np.array.
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
