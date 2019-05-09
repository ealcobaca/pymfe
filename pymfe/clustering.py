"""A module dedicated to the extraction of Clustering Metafeatures.
"""
import typing as t
import itertools

import numpy as np
import scipy.spatial.distance
import sklearn.metrics
import sklearn.neighbors


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
                                   N: np.ndarray,
                                   y: t.Optional[np.ndarray] = None,
                                   dist_metric: str = "euclidean",
                                   classes: t.Optional[np.ndarray] = None,
                                   **kwargs) -> t.Dict[str, t.Any]:
        """Precompute distance metrics between instances.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical attributes from fitted data.

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
                * ``pairwise_norm_interclass_dist`` (:obj:`np.ndarray`):
                  normalized distance between each distinct pair of
                  instances of different classes.

                * ``pairwise_intraclass_dists`` (:obj:`np.ndarray`):
                  distance between each distinct pair of instances of
                  the same class.

                * ``intraclass_dists`` (:obj:`np.ndarray`): the distance
                  between the fartest pair of instances of the same class.
        """
        precomp_vals = {}

        if y is not None and not {
                "pairwise_norm_interclass_dist",
                "pairwise_intraclass_dists",
                "intraclass_dists"}.issubset(kwargs):
            precomp_vals["pairwise_norm_interclass_dist"] = (
                MFEClustering._pairwise_norm_interclass_dist(
                    N=N, y=y, dist_metric=dist_metric, classes=classes))

            precomp_vals["pairwise_intraclass_dists"] = (
                MFEClustering._all_intraclass_dists(
                    N=N,
                    y=y,
                    dist_metric=dist_metric,
                    classes=classes,
                    get_max_dist=False))

            precomp_vals["intraclass_dists"] = (
                precomp_vals["pairwise_intraclass_dists"].max(axis=1))

        return precomp_vals

    @classmethod
    def _normalized_interclass_dist(
            cls,
            group_inst_a: np.ndarray,
            group_inst_b: np.ndarray,
            dist_metric: str = "euclidean",
    ) -> np.ndarray:
        """Calculate the distance between instances of different classes.

        The distance is normalized by the number of distinct pairs
        between ``group_inst_a`` and ``group_inst_b``.
        """
        norm_interclass_dist = scipy.spatial.distance.cdist(
            group_inst_a,
            group_inst_b,
            metric=dist_metric)

        return norm_interclass_dist / norm_interclass_dist.size

    @classmethod
    def _pairwise_norm_interclass_dist(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            dist_metric: str = "euclidean",
            classes: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate all pairwise normalized interclass distances."""
        if classes is None:
            classes = np.unique(y)

        interclass_dists = np.array([
            MFEClustering._normalized_interclass_dist(
                N[y == class_a, :],
                N[y == class_b, :],
                dist_metric=dist_metric)
            for class_a, class_b in itertools.combinations(classes, 2)
        ])

        return interclass_dists

    @classmethod
    def _intraclass_dists(cls,
                          instances: np.ndarray,
                          dist_metric: str = "euclidean",
                          get_max_dist: bool = True) -> float:
        """Calculate the intraclass distance of the given instances.

        The intraclass is the maximum distance between two distinct
        instances of the same class. If ``get_max`` is false, then
        all distances are returned instead.
        """
        intraclass_dists = scipy.spatial.distance.pdist(
            instances, metric=dist_metric)

        return intraclass_dists.max() if get_max_dist else intraclass_dists

    @classmethod
    def _all_intraclass_dists(cls,
                              N: np.ndarray,
                              y: np.ndarray,
                              dist_metric: str = "euclidean",
                              classes: t.Optional[np.ndarray] = None,
                              get_max_dist: bool = True) -> np.ndarray:
        """Calculate all intraclass (internal to a class) distances."""
        if classes is None:
            classes = np.unique(y)

        intraclass_dists = np.array([
            MFEClustering._intraclass_dists(
                N[y == cur_class, :],
                dist_metric=dist_metric,
                get_max_dist=get_max_dist) for cur_class in classes
        ])

        return intraclass_dists

    @classmethod
    def ft_dunn_index(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            dist_metric: str = "euclidean",
            classes: t.Optional[np.ndarray] = None,
            intraclass_dists: t.Optional[np.ndarray] = None,
            pairwise_norm_interclass_dist: t.Optional[np.ndarray] = None,
            epsilon: float = 1.0e-8,
    ) -> float:
        """Calculate the Dunn Index.

        Metric range is 0 (inclusive) and infinity.

        Parameters
        ----------
            dist_metric : :obj:`str`, optional
                The distance metric used to calculate the distances between
                instances. Check `scipydoc`_ for a full list of valid distance
                metrics. If precomputation in clustering metafeatures is
                enabled, then this parameter takes no effect.

            classes : :obj:`np.ndarray`, optional
                Distinct classes in ``y``. Used to exploit precomputations.

            intraclass_dists : :obj:`np.ndarray`, optional
                Distance between the fartest pair of instances in the same
                class, for each class. Used to exploit precomputations.

            pairwise_norm_interclass_dists : :obj:`np.ndarray`, optional
                Normalized pairwise distances between instances of different
                classes.

            epsilon : :obj:`float`, optional
                A tiny value used to prevent division by zero.

        Returns
        -------
        :obj:`float`
            Dunn index for given parameters.

        Notes
        -----
            .. _scipydoc: :obj:`scipy.spatial.distance` documentation.
        """
        if pairwise_norm_interclass_dist is None:
            pairwise_norm_interclass_dist = (
                MFEClustering._pairwise_norm_interclass_dist(
                    N=N, y=y, dist_metric=dist_metric, classes=classes))

        if intraclass_dists is None:
            intraclass_dists = MFEClustering._all_intraclass_dists(
                N=N, y=y, dist_metric=dist_metric, classes=classes).max()

        vdu = (pairwise_norm_interclass_dist.min() /
               (intraclass_dists.max() + epsilon))

        return vdu

    @classmethod
    def ft_davies_bouldin_index(cls, N: np.ndarray, y: np.ndarray) -> float:
        """Calculate the Davies and Bouldin Index.

        Metric range is 0 (inclusive) and infinity.

        Check `dbindex`_ for more information.

        Notes
        -----
            .. _dbindex: :obj:``sklearn.metrics.davies_bouldin_score``
                documentation.
        """
        return sklearn.metrics.davies_bouldin_score(X=N, labels=y)

    @classmethod
    def ft_int(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            dist_metric: str = "euclidean",
            classes: t.Optional[np.ndarray] = None,
            pairwise_norm_interclass_dist: t.Optional[np.ndarray] = None,
    ) -> float:
        """Calculate the INT index.

        Metric range is 0 (inclusive) and infinity.

        Parameters
        ----------
            dist_metric : :obj:`str`, optional
                The distance metric used to calculate the distances between
                instances. Check `scipydoc`_ for a full list of valid distance
                metrics. If precomputation in clustering metafeatures is
                enabled, then this parameter takes no effect.

            classes : :obj:`np.ndarray`, optional
                Distinct classes in ``y``. Used to exploit precomputations.

            pairwise_norm_interclass_dists : :obj:`np.ndarray`, optional
                Normalized pairwise distances between instances of different
                classes. Used to exploit precomputations.

        Returns
        -------
        :obj:`float`
            INT index.

        Notes
        -----
            .. _scipydoc: :obj:`scipy.spatial.distance` documentation.
        """
        if classes is None:
            classes = np.unique(y)

        class_num = classes.size

        if class_num == 1:
            return np.nan

        if pairwise_norm_interclass_dist is None:
            pairwise_norm_interclass_dist = (
                MFEClustering._pairwise_norm_interclass_dist(
                    N=N, y=y, dist_metric=dist_metric, classes=classes))

        norm_factor = 2.0 / (class_num * (class_num - 1.0))

        return pairwise_norm_interclass_dist.sum() * norm_factor

    @classmethod
    def ft_connectivity(cls,
                        N: np.ndarray,
                        y: np.ndarray,
                        dist_metric: str = "euclidean",
                        n_neighbors: t.Optional[int] = None,
                        class_freqs: t.Optional[np.ndarray] = None) -> float:
        """Calculate the connectivity Index.

        Metric range is 0 (inclusive) to infinity.

        Parameters
        ----------
            dist_metric : :obj:`str`, optional
                The distance metric used to calculate the distances between
                instances. Check `distmetric`_ for a full list of valid
                distance metrics.

            n_neighbors : :obj:`int`, optional
                Numbers of considered neighbors. If not given, the default
                value will be the square root of the number of instances in
                the minority class (i.e., the square root of the smallest
                (absolute) frequency of classes in ``y``).

            class_freqs : :obj:`np.ndarray`, optional
                Class (absolute) frequencies. Used to exploit precomputations.

        Returns
        -------
        :obj:`float`
            Connectivity Index.

        Notes
        -----
            .. _distmetric: :obj:`sklearn.neighbors.DistanceMetric`
                documentation.
        """
        if class_freqs is None:
            _, class_freqs = np.unique(y, return_counts=True)

        if n_neighbors is None:
            n_neighbors = int(np.sqrt(class_freqs.min()))

        model = sklearn.neighbors.KDTree(N, metric=dist_metric)

        # Note: skip the first column because it's always the
        # instance itself
        nearest_neig = model.query(
            N, k=n_neighbors + 1, return_distance=False)[:, 1:]

        con_index = np.array([
            sum([
                1.0 / proximity_ind
                for proximity_ind, neig_inst_ind in enumerate(
                    nearest_neig[cur_inst_ind, :], 1)
                if y[neig_inst_ind] == y[cur_inst_ind]
            ]) for cur_inst_ind in np.arange(y.size)
        ]).sum()

        return con_index

    @classmethod
    def ft_silhouette(cls,
                      N: np.ndarray,
                      y: np.ndarray,
                      dist_metric: str = "euclidean",
                      sample_size: t.Optional[int] = None,
                      random_state: t.Optional[int] = None) -> float:
        """Calculate the silhouette value.

        Metric range is -1 to +1 (both inclusive).

        Check `silhouette`_ for more information.

        Parameters
        ----------
            dist_metric : :obj:`str`, optional
                The distance metric used to calculate the distances between
                instances. Check `distmetric`_ for a full list of valid
                distance metrics.

            sample_size : :obj:`int`, optional
                Sample size used to compute the silhouette coefficient. If
                None is used, then all data is used.

            random_state : :obj:`int`, optional
                Used if ``sample_size`` is not None. Random seed to be
                used while sampling the data.

        Returns
        -------
        :obj:`float`
            Silhouette value.

        Notes
        -----
            .. _silhouette: :obj:`sklearn.metrics.silhouette_score`
                documentation.
            .. _distmetric: :obj:`sklearn.neighbors.DistanceMetric`
                documentation.
        """
        silhouette = sklearn.metrics.silhouette_score(
            X=N,
            labels=y,
            metric=dist_metric,
            sample_size=sample_size,
            random_state=random_state)

        return silhouette

    @classmethod
    def ft_goodman_kruskal_gamma(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            dist_metric: str = "euclidean",
            classes: t.Optional[np.ndarray] = None,
            pairwise_intraclass_dists: t.Optional[np.ndarray] = None
            ) -> np.ndarray:
        """Goodman and Kruskal's Gamma rank correlation.

        The range value is -1 and +1 (both inclusive).

        Parameters
        ----------
            dist_metric : :obj:`str`, optional
                The distance metric used to calculate the distances between
                instances. Check `scipydoc`_ for a full list of valid distance
                metrics. If precomputation in clustering metafeatures is
                enabled, then this parameter takes no effect.

            classes : :obj:`np.ndarray`, optional
                Distinct classes in ``y``. Used to exploit precomputations.

            pairwise_intraclass_dists : :obj:`np.ndarray`, optional
                Distance between each distinct pair of instances of
                the same class. Used to exploit precomputations.

        Returns
        -------
        :obj:`float`
            Goodman and Kruskal Gamma

        Notes
        -----
            .. _scipydoc: :obj:`scipy.spatial.distance` documentation.
        """
        if classes is None:
            classes = np.unique(y)

        if pairwise_intraclass_dists is None:
            pairwise_intraclass_dists = MFEClustering._all_intraclass_dists(
                N=N,
                y=y,
                dist_metric=dist_metric,
                classes=classes,
                get_max_dist=False)

        gk_gamma = []

        for class_a, class_b in itertools.combinations(
                np.arange(classes.size), 2):
            pdists_a = pairwise_intraclass_dists[class_a]
            pdists_b = pairwise_intraclass_dists[class_b]

            pairs_concordant = 0
            pairs_reversed = 0

            size = min(pdists_a.size, pdists_b.size)

            for i, j in itertools.combinations(np.arange(size), 2):
                sign_a = np.sign(pdists_a[i] - pdists_a[j])
                sign_b = np.sign(pdists_b[i] - pdists_b[j])

                if sign_a * sign_b:
                    pairs_concordant += sign_a == sign_b
                    pairs_reversed += sign_a == -sign_b

            gk_gamma.append((pairs_concordant - pairs_reversed) /
                            (pairs_concordant + pairs_reversed))

        return np.array(gk_gamma)

    @classmethod
    def ft_point_biserial(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            dist_metric: str = "euclidean",
    ) -> float:
        """Pearson Correlation between class matching and instance distances.

        The measure interval is -1 and +1 (inclusive).

        Parameters
        ----------
            dist_metric : :obj:`str`, optional
                The distance metric used to calculate the distances between
                instances. Check `scipydoc`_ for a full list of valid distance
                metrics.

        Returns
        -------
        :obj:`float`
            Point Bisseral coefficient.

        Notes
        -----
            .. _scipydoc: :obj:`scipy.spatial.distance` documentation.
        """
        inst_dists = scipy.spatial.distance.pdist(
            X=N, metric=dist_metric)

        inst_matching_classes = np.array([
            inst_class_a == inst_class_b
            for inst_class_a, inst_class_b in itertools.combinations(y, 2)
        ])

        correlation, _ = scipy.stats.pointbiserialr(
            x=inst_matching_classes, y=inst_dists)

        return correlation

    @classmethod
    def ft_hubert_levin_index(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            dist_metric: str = "euclidean",
            classes: t.Optional[np.ndarray] = None,
            class_freqs: t.Optional[np.ndarray] = None,
            pairwise_intraclass_dists: t.Optional[np.ndarray] = None
            ) -> np.ndarray:
        """Hubert and Levin index.

        The metric range is 0 and 1 (both inclusive).

        Parameters
        ----------
            dist_metric : :obj:`str`, optional
                The distance metric used to calculate the distances between
                instances. Check `scipydoc`_ for a full list of valid distance
                metrics.

            classes : :obj:`np.ndarray`, optional
                Distinct classes in ``y``. Used to exploit precomputations.

            class_freqs : :obj:`np.ndarray`, optional
                Class (absolute) frequencies. Used to exploit precomputations.

            pairwise_intraclass_dists : :obj:`np.ndarray`, optional
                Distance between each distinct pair of instances of
                the same class. Used to exploit precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Hubert and Levin index for each distinct class.

        Notes
        -----
            .. _scipydoc: :obj:`scipy.spatial.distance` documentation.
        """
        if classes is None or class_freqs is None:
            classes, class_freqs = np.unique(y, return_counts=True)

        sorted_pairwise_dist = scipy.spatial.distance.pdist(
            X=N, metric=dist_metric)

        sorted_pairwise_dist.sort()

        if pairwise_intraclass_dists is None:
            pairwise_intraclass_dists = MFEClustering._all_intraclass_dists(
                N=N,
                y=y,
                dist_metric=dist_metric,
                classes=classes,
                get_max_dist=False)

        sum_intracl_dists = pairwise_intraclass_dists.sum(axis=1)

        c_indexes = np.zeros(classes.size)

        for i in np.arange(classes.size):
            cl_pair_num = class_freqs[i] * (class_freqs[i] - 1) // 2

            sum_d_rank_least = sorted_pairwise_dist[:cl_pair_num].sum()
            sum_d_rank_greater = sorted_pairwise_dist[-cl_pair_num:].sum()

            c_indexes[i] = ((sum_intracl_dists[i] - sum_d_rank_least) /
                            (sum_d_rank_greater - sum_d_rank_least))

        return c_indexes

    @classmethod
    def ft_calinski_harabaz_index(cls, N: np.ndarray, y: np.ndarray) -> float:
        """Calinski and Harabaz index.

        Check `cahascore`_ for more information.

        Returns
        -------
        :obj:`float`
            Calinski-Harabanz index.

        Notes
        -----
            .. _cahascore: ``sklearn.metrics.calinski_harabaz_score``
                documentation.
        """
        return sklearn.metrics.calinski_harabaz_score(X=N, labels=y)

    @classmethod
    def ft_mn(cls, N: np.ndarray, y: np.ndarray) -> float:
        """Related to multivariate normality.

        For its estimation, the instances are initially transformed
        into values of Hotelling’s T**2 statistics.

        Parameters
        ----------

        Returns
        -------
        """

    @classmethod
    def ft_sk(cls, N: np.ndarray, y: np.ndarray) -> float:
        """It returns the skewness of the T**2 vector.

        Parameters
        ----------

        Returns
        -------
        """

    @classmethod
    def ft_po(cls, N: np.ndarray, y: np.ndarray) -> float:
        """Percentage of outliers using the T**2 vector.

        Values of T**2 more distant than two standard deviations from
        the mean are considered outliers.

        Parameters
        ----------

        Returns
        -------
        """

    @classmethod
    def ft_nre(cls, N: np.ndarray, y: np.ndarray) -> float:
        """Normalized relative entropy.

        An indicator of uniformity distributed of instances among clusters.

        Parameters
        ----------

        Returns
        -------
        """

    @classmethod
    def ft_sc(cls, N: np.ndarray, y: np.ndarray, size: int = 15) -> float:
        """Number of clusters with size smaller than ``size``.

        Parameters
        ----------

        Returns
        -------
        """

    @classmethod
    def ft_cm(cls, N: np.ndarray, y: np.ndarray) -> float:
        """Sum of distances of items to corresponding cluster representatives.

        Parameters
        ----------

        Returns
        -------
        """

    @classmethod
    def ft_si(cls, N: np.ndarray, y: np.ndarray) -> float:
        """Global silhouette index.

        It checks whether the current cluster of every instance is more
        appropriate than the neighboring cluster.

        Parameters
        ----------

        Returns
        -------
        """

    @classmethod
    def ft_aic(cls, N: np.ndarray, y: np.ndarray) -> float:
        """Internal cluster evaluation measure.

        Parameters
        ----------

        Returns
        -------
        """

    @classmethod
    def ft_bic(cls, N: np.ndarray, y: np.ndarray) -> float:
        """Bayesian Information Criterion.

        This measure is based on maximized value of the likelihood function
        of the model.

        Parameters
        ----------

        Returns
        -------
        """

    @classmethod
    def ft_xb(cls, N: np.ndarray, y: np.ndarray) -> float:
        """Ratio of overall deviation to cluster separation.

        Parameters
        ----------

        Returns
        -------
        """

    @classmethod
    def ft_cn(cls, N: np.ndarray, y: np.ndarray) -> float:
        """Measures whether neighboring items are in the same cluster.

        Parameters
        ----------

        Returns
        -------
        """


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()

    ans = MFEClustering.ft_goodman_kruskal_gamma(iris.data, iris.target)
    print(ans)
