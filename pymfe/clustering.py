"""A module dedicated to the extraction of Clustering Metafeatures.
"""
import typing as t

import numpy as np
import scipy.spatial.distance
import itertools
import sklearn.metrics


class MFEClustering:
    """Keep methods for metafeatures of ``Clustering`` group.

    The convention adopted for metafeature extraction related methods is to
    always start with ``ft_`` prefix to allow automatic method detection. This
    prefix is predefined within ``_internal`` module.

    All method signature follows the conventions and restrictions listed below:

    1. For independent attribute data, ``X`` means ``every type of
       attribute``, ``N`` means ``Numeric attributes only`` and ``C`` stands
       for ``Categorical attributes only``. It is important to note that the
       categorical attribute sets between ``X`` and ``C`` and the numerical
       attribute sets between ``X`` and ``N`` may differ due to data
       transformations, performed while fitting data into MFE model,
       enabled by, respectively, ``transform_num`` and ``transform_cat``
       arguments from ``fit`` (MFE method).

    2. Only arguments in MFE ``_custom_args_ft`` attribute (set up inside
       ``fit`` method) are allowed to be required method arguments. All other
       arguments must be strictly optional (i.e., has a predefined default
       value).

    3. The initial assumption is that the user can change any optional
       argument, without any previous verification of argument value or its
       type, via kwargs argument of ``extract`` method of MFE class.

    4. The return value of all feature extraction methods should be a single
       value or a generic Sequence (preferably a :obj:`np.ndarray`)
       type with numeric values.

    There is another type of method adopted for automatic detection. It is
    adopted the prefix ``precompute_`` for automatic detection of these
    methods. These methods run while fitting some data into an MFE model
    automatically, and their objective is to precompute some common value
    shared between more than one feature extraction method. This strategy is a
    trade-off between more system memory consumption and speeds up of feature
    extraction. Their return value must always be a dictionary whose keys are
    possible extra arguments for both feature extraction methods and other
    precomputation methods. Note that there is a share of precomputed values
    between all valid feature-extraction modules (e.g., ``class_freqs``
    computed in module ``statistical`` can freely be used for any
    precomputation or feature extraction method of module ``landmarking``).
    """

    @classmethod
    def precompute_clustering_class(cls,
                                    y: t.Optional[np.ndarray] = None,
                                    **kwargs) -> t.Dict[str, t.Any]:
        """Precompute distinct classes and its frequencies from ``y``.

        Parameters
        ----------
        y : :obj:`np.ndarray`, optional
            Target attribute from fitted data.

        **kwargs
            Additional arguments. May have previously precomputed before
            this method from other precomputed methods, so they can help
            speed up this precomputation.

        Returns
        -------
        :obj:`dict`

            The following precomputed items are returned:
                * ``classes`` (:obj:`np.ndarray`):  distinct classes of
                  ``y``, if ``y`` is not :obj:`NoneType`.
                * ``class_freqs`` (:obj:`np.ndarray`): class frequencies of
                  ``y``, if ``y`` is not :obj:`NoneType`.
        """
        precomp_vals = {}

        if y is not None and not {"classes", "class_freqs"}.issubset(kwargs):
            classes, class_freqs = np.unique(y, return_counts=True)

            precomp_vals["classes"] = classes
            precomp_vals["class_freqs"] = class_freqs

        return precomp_vals

    @classmethod
    def precompute_group_distances(cls,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   dist_metric: str = "gower",
                                   classes: t.Optional[np.ndarray] = None,
                                   **kwargs) -> t.Dict[str, t.Any]:
        """."""
        precomp_vals = {}

        if not {"pairwise_norm_interclass_dist", "intraclass_dists"
                }.issubset(kwargs):
            precomp_vals["pairwise_norm_interclass_dist"] = (
                MFEClustering._pairwise_norm_interclass_dist(
                    X=X, y=y, dist_metric=dist_metric, classes=classes))

            precomp_vals["intraclass_dists"] = (
                MFEClustering._all_intraclass_dists(
                    X=X, y=y, dist_metric=dist_metric, classes=classes))

        return precomp_vals

    @classmethod
    def _gower_dist(cls,
                    vec_x: np.ndarray,
                    vec_y: np.ndarray,
                    epsilon: float = 1.0e-8) -> float:
        """Calculate the Gower distance between ``vec_x`` and ``vec_y``."""
        return np.linalg.norm(vec_x - vec_y, 2)

    @classmethod
    def _normalized_interclass_dist(
            cls,
            group_inst_a: np.ndarray,
            group_inst_b: np.ndarray,
            dist_metric: str = "gower",
    ) -> np.ndarray:
        """Calculate the distance between instances of different classes.

        The distance is normalized by the multiplication of the number of
        instances of each groups.
        """
        norm_interclass_dist = scipy.spatial.distance.cdist(
            group_inst_a,
            group_inst_b,
            metric=(dist_metric
                    if dist_metric != "gower" else MFEClustering._gower_dist))

        # Note: norm_interclass_dist.size =
        #   group_inst_a.size * group_inst_b.size
        return norm_interclass_dist / norm_interclass_dist.size

    @classmethod
    def _pairwise_norm_interclass_dist(
            cls,
            X: np.ndarray,
            y: np.ndarray,
            dist_metric: str = "gower",
            classes: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate all pairwise normalized interclass distances."""
        if classes is None:
            classes = np.unique(y)

        interclass_dists = np.array([
            MFEClustering._normalized_interclass_dist(
                X[y == class_a, :],
                X[y == class_b, :],
                dist_metric=dist_metric)
            for class_a, class_b in itertools.combinations(classes, 2)
        ])

        return interclass_dists

    @classmethod
    def _intraclass_dists(cls,
                          instances: np.ndarray,
                          dist_metric: str = "gower") -> float:
        """Calculate the intraclass distance of the given instances.

        The intraclass is the maximum distance between two distinct
        instances of the same class.
        """
        intraclass_dists = scipy.spatial.distance.pdist(
            instances,
            metric=(dist_metric
                    if dist_metric != "gower" else MFEClustering._gower_dist))

        return intraclass_dists.max()

    @classmethod
    def _all_intraclass_dists(
            cls,
            X: np.ndarray,
            y: np.ndarray,
            dist_metric: str = "gower",
            classes: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate all intraclass (internal to a class) distances."""
        if classes is None:
            classes = np.unique(y)

        intraclass_dists = np.array([
            MFEClustering._intraclass_dists(
                X[y == cur_class, :], dist_metric=dist_metric)
            for cur_class in classes
        ])

        return intraclass_dists

    @classmethod
    def ft_vdu(
            cls,
            X: np.ndarray,
            y: np.ndarray,
            dist_metric: str = "gower",
            classes: t.Optional[np.ndarray] = None,
            intraclass_dists: t.Optional[np.ndarray] = None,
            pairwise_norm_interclass_dist: t.Optional[np.ndarray] = None,
            epsilon: float = 1.0e-8,
    ) -> float:
        """
        Parameters
        ----------

        Returns
        -------
        """
        if pairwise_norm_interclass_dist is None:
            pairwise_norm_interclass_dist = (
                MFEClustering._pairwise_norm_interclass_dist(
                    X=X, y=y, dist_metric=dist_metric, classes=classes))

        if intraclass_dists is None:
            intraclass_dists = MFEClustering._all_intraclass_dists(
                X=X, y=y, dist_metric=dist_metric, classes=classes).max()

        vdu = (pairwise_norm_interclass_dist.min() /
               (intraclass_dists.max() + epsilon))

        return vdu

    @classmethod
    def ft_vdb(cls) -> float:
        """
        Parameters
        ----------

        Returns
        -------
        """

    @classmethod
    def ft_int(
            cls,
            X: np.ndarray,
            y: np.ndarray,
            dist_metric: str = "gower",
            classes: t.Optional[np.ndarray] = None,
            pairwise_norm_interclass_dist: t.Optional[np.ndarray] = None,
    ) -> float:
        """
        Parameters
        ----------

        Returns
        -------
        """
        if classes is None:
            classes = np.unique(y)

        class_num = classes.size

        if class_num == 1:
            return np.nan

        if pairwise_norm_interclass_dist is None:
            pairwise_norm_interclass_dist = (
                MFEClustering._pairwise_norm_interclass_dist(
                    X=X, y=y, dist_metric=dist_metric, classes=classes))

        norm_factor = 2.0 / (class_num * (class_num - 1.0))

        return pairwise_norm_interclass_dist.sum() * norm_factor

    @classmethod
    def ft_con(cls) -> float:
        """
        Parameters
        ----------

        Returns
        -------
        """

    @classmethod
    def ft_silhouette(cls,
                      X: np.ndarray,
                      y: np.ndarray,
                      dist_metric: str = "gower",
                      sample_size: t.Optional[int] = None,
                      random_state: t.Optional[int] = None) -> float:
        """
        Parameters
        ----------

        Returns
        -------
        """
        silhouette = sklearn.metrics.silhouette_score(
            X=X,
            labels=y,
            metric=(dist_metric
                    if dist_metric != "gower" else MFEClustering._gower_dist),
            sample_size=sample_size,
            random_state=random_state)

        return silhouette

    @classmethod
    def ft_goodman_kruskal_gamma(
            cls,
            X: np.ndarray,
            y: np.ndarray,
            dist_metric: str = "gower",
            classes: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Goodman and Kruskal's Gamma rank correlation.

        The range value is [-1, 1].

        TO BE FIXED.

        Parameters
        ----------

        Returns
        -------
        """
        if classes is None:
            classes = np.unique(y)

        pairwise_intraclass_dists = {
            cur_class: scipy.spatial.distance.pdist(
                X[y == cur_class, :],
                metric=(dist_metric if dist_metric != "gower" else
                        MFEClustering._gower_dist))
            for cur_class in classes
        }

        gk_gamma = []

        for class_a, class_b in itertools.combinations(classes, 2):
            pairwise_intraclass_dists_a = pairwise_intraclass_dists[class_a]
            pairwise_intraclass_dists_b = pairwise_intraclass_dists[class_b]

            pairs_concordant = 0
            for dist_a in pairwise_intraclass_dists_a:
                for dist_b in pairwise_intraclass_dists_b:
                    pairs_concordant += 1 if dist_a < dist_b else -1

            gk_gamma.append(
                ((pairs_concordant) / (pairwise_intraclass_dists_a.size *
                                       pairwise_intraclass_dists_b.size)))

        return np.array(gk_gamma)

    @classmethod
    def ft_point_biserial(
            cls,
            X: np.ndarray,
            y: np.ndarray,
            dist_metric: str = "gower",
    ) -> float:
        """Pearson Correlation between class matching and instance distances.

        The measure interval is [-1, 1].

        Parameters
        ----------

        Returns
        -------
        """
        inst_dists = scipy.spatial.distance.pdist(
            X=X,
            metric=(dist_metric
                    if dist_metric != "gower" else MFEClustering._gower_dist))

        inst_matching_classes = np.array([
            inst_class_a == inst_class_b
            for inst_class_a, inst_class_b in itertools.combinations(y, 2)
        ])

        correlation, _ = scipy.stats.pointbiserialr(
            x=inst_matching_classes, y=inst_dists)

        return correlation

    @classmethod
    def ft_hl(cls) -> float:
        """
        Parameters
        ----------

        Returns
        -------
        """

    @classmethod
    def ft_ch(cls) -> float:
        """
        Parameters
        ----------

        Returns
        -------
        """


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    """
    instances = iris.data[:5, :]

    intraclass_dists = scipy.spatial.distance.pdist(
        instances,
        metric=MFEClustering._gower_dist)

    print(intraclass_dists)
    """
    data = np.array([
        2,
        8,
        5,
        4,
        2,
        6,
        1,
        4,
        5,
        7,
        4,
        3,
        9,
        4,
        3,
        1,
        7,
        2,
        5,
        6,
        8,
        3,
    ]).reshape(-1, 1)

    y = np.array([0] * 11 + [1] * 11)

    gk_gamma = MFEClustering.ft_point_biserial(data, y)

    print(gk_gamma)
