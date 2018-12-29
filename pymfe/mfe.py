"""Main module for extracting metafeatures from datasets.

Todo:
    - Improve documentation.
    - Implement MFE class.
"""
import collections


class MFE:
    """Core class for metafeature extraction."""
    VALID_GROUPS = (
        "landmarking",
        "general",
        "statistical",
        "model-based",
        "info-theory",
    )

    def __init__(self, groups="all", features="all"):
        """To do this documentation."""

        self.groups = MFE._process_groups(groups)
        self.features = features

    @classmethod
    def _process_groups(cls, groups):
        """Check if "groups" parameter is correct.

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
            AttributeError: if `groups` is not "all", a Iterable
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
                return MFE.VALID_GROUPS

            if groups in MFE.VALID_GROUPS:
                return (groups,)

            unknown_groups = {groups}

        elif isinstance(groups, collections.Iterable):
            groups = set(map(str.lower, groups))

            if "all" in groups:
                return MFE.VALID_GROUPS

            if groups.issubset(MFE.VALID_GROUPS):
                return tuple(groups)

            unknown_groups = groups.difference(MFE.VALID_GROUPS)

        if unknown_groups is not None:
            raise ValueError("Unknown groups: {0}".format(unknown_groups))

        raise AttributeError('"Groups" parameter type is not consistent.')
