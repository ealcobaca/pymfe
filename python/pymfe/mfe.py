"""Main module for extracting metafeatures from datasets.

Todo:
    * Improve documentation.
    * Implement MFE class.
"""
import _internal


class MFE:
    """Core class for metafeature extraction."""

    def __init__(self, groups="all", features="all", summary="all"):
        """To do this documentation."""

        self.groups = _internal.process_groups(groups)
        self.features = _internal.process_features(features)
        self.summary = _internal.process_summary(summary)
        self.X = None
        self.y = None

    def fit(self, X, y, splits=None):
        """Fits dataset into the MFE model.

        Args:
            X (:obj:`list` or :obj:`numpy.array`): predictive attributes of
                the dataset.
            y (:obj:`list` or :obj:`numpy.array`): target attributes of the
                dataset.
            splits (:obj:`Iterable` of :obj:`ints`, optional): iterable which
                contains K-Fold Cross Validation index splits to use mainly in
                landmarking metafeatures. If not given, each metafeature will
                be extracted a single time.

        Raises:
            ValueError: if number of rows of X and y does not match.
            TypeError: if X or y (or both) is neither a :obj:`list` or
                a :obj:`np.array` object.
        """

        self.X, self.y = _internal.check_data(X, y)

    def extract(self):
        """Extracts metafeatures from fitted dataset."""
