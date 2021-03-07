"""A module dedicated to the extraction of clustering metafeatures.
"""
import typing as t
import itertools

import numpy as np
import scipy.spatial.distance
import sklearn.metrics
import sklearn.neighbors

from pymfe import _utils


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
       value or a generic List (preferably a :obj:`np.ndarray`)
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
    def precompute_clustering_class(
        cls, y: t.Optional[np.ndarray] = None, **kwargs
    ) -> t.Dict[str, t.Any]:
        """Precompute distinct classes and its frequencies from ``y``.

        Parameters
        ----------
        y : :obj:`np.ndarray`, optional
            Instance cluster index (or target attribute).

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
                * ``cls_inds`` (:obj:`np.ndarray`): Boolean array which
                  indicates whether each example belongs to each class. The
                  rows represents the distinct classes, and the instances
                  are represented by the columns.
        """
        precomp_vals = {}

        if y is not None and not {"classes", "class_freqs"}.issubset(kwargs):
            classes, class_freqs = np.unique(y, return_counts=True)

            precomp_vals["classes"] = classes
            precomp_vals["class_freqs"] = class_freqs

        classes = kwargs.get("classes", precomp_vals.get("classes"))

        if y is not None and "cls_inds" not in kwargs:
            cls_inds = _utils.calc_cls_inds(y, classes)
            precomp_vals["cls_inds"] = cls_inds

        return precomp_vals

    @classmethod
    def precompute_group_distances(
        cls,
        N: np.ndarray,
        y: t.Optional[np.ndarray] = None,
        dist_metric: str = "euclidean",
        classes: t.Optional[np.ndarray] = None,
        **kwargs
    ) -> t.Dict[str, t.Any]:
        """Precompute distance metrics between instances.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`, optional
            Instance cluster index (or target attribute).

        dist_metric : str, optional
            The distance metric used to calculate the distances between
            instances. Check :obj:`sklearn.neighbors.DistanceMetric`
            documentation for a full list of valid distance metrics.

        classes : :obj:`np.ndarray`, optional
            Distinct classes in ``y``. Used to exploit precomputations.

        **kwargs
            Additional arguments. May have previously precomputed before
            this method from other precomputed methods, so they can help
            speed up this precomputation.

        Returns
        -------
        :obj:`dict`
            The following precomputed items are returned:
                * ``pairwise_norm_intercls_dist`` (:obj:`np.ndarray`):
                  normalized distance between each distinct pair of
                  instances of different classes.
                * ``pairwise_intracls_dists`` (:obj:`np.ndarray`):
                  distance between each distinct pair of instances of
                  the same class.
                * ``intracls_dists`` (:obj:`np.ndarray`): the distance
                  between the fartest pair of instances of the same class.

        The following precomputed items are necessary and are also
        returned, if still not previously precomputed:
            * ``classes`` (:obj:`np.ndarray`):  distinct classes of
              ``y``, if ``y`` is not :obj:`NoneType`.
            * ``class_freqs`` (:obj:`np.ndarray`): class frequencies of
              ``y``, if ``y`` is not :obj:`NoneType`.
            * ``cls_inds`` (:obj:`np.ndarray`): Boolean array which
              indicates whether each example belongs to each class. The
              rows represents the distinct classes, and the instances
              are represented by the columns.
        """
        precomp_vals = {}

        if (
            N is not None
            and y is not None
            and not {
                "pairwise_norm_intercls_dist",
                "pairwise_intracls_dists",
                "intracls_dists",
            }.issubset(kwargs)
        ):
            cls_inds = kwargs.get("cls_inds")

            if cls_inds is None:
                new_vals = cls.precompute_clustering_class(**kwargs)
                cls_inds = new_vals["cls_inds"]
                classes = new_vals["classes"]
                precomp_vals.update(new_vals)

            precomp_vals[
                "pairwise_norm_intercls_dist"
            ] = cls._calc_pwise_norm_intercls_dist(
                N=N,
                y=y,
                dist_metric=dist_metric,
                classes=classes,
                cls_inds=cls_inds,
            )

            precomp_vals[
                "pairwise_intracls_dists"
            ] = cls._calc_all_intracls_dists(
                N=N,
                y=y,
                dist_metric=dist_metric,
                cls_inds=cls_inds,
                classes=classes,
                get_max_dist=False,
            )

            if precomp_vals["pairwise_intracls_dists"].ndim == 2:
                precomp_vals["intracls_dists"] = precomp_vals[
                    "pairwise_intracls_dists"
                ].max(axis=1)

            else:
                precomp_vals["intracls_dists"] = np.array(
                    [
                        np.max(class_arr)
                        for class_arr in precomp_vals[
                            "pairwise_intracls_dists"
                        ]
                    ]
                )

        return precomp_vals

    @classmethod
    def precompute_nearest_neighbors(
        cls,
        N: np.ndarray,
        y: t.Optional[np.ndarray] = None,
        n_neighbors: t.Optional[int] = None,
        dist_metric: str = "euclidean",
        **kwargs
    ) -> t.Dict[str, t.Any]:
        """Precompute the ``n_neighbors`` Nearest Neighbors of every instance.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`, optional
            Instance cluster index (or target attribute).

        n_neighbors : int, optional
            Number of nearest neighbors returned for each instance.

        dist_metric : str, optional
            The distance metric used to calculate the distances between
            instances. Check :obj:`sklearn.neighbors.DistanceMetric`
            documentation for a full list of valid distance metrics.

        **kwargs
            Additional arguments. May have previously precomputed before
            this method from other precomputed methods, so they can help
            speed up this precomputation.

        Returns
        -------
        :obj:`dict`
            The following precomputed items are returned:
                * ``pairwise_intracls_dists`` (:obj:`np.ndarray`):
                  distance between each distinct pair of instances of
                  the same class.
        """
        precomp_vals = {}

        if (
            N is not None
            and y is not None
            and not {"nearest_neighbors"}.issubset(kwargs)
        ):
            class_freqs = kwargs.get("class_freqs")

            if class_freqs is None:
                _, class_freqs = np.unique(y, return_counts=True)

            if n_neighbors is None:
                n_neighbors = int(np.sqrt(class_freqs.min()))

            precomp_vals["nearest_neighbors"] = cls._get_nearest_neighbors(
                N=N, n_neighbors=n_neighbors, dist_metric=dist_metric
            )

        return precomp_vals

    @classmethod
    def precompute_class_representatives(
        cls,
        N: np.ndarray,
        y: t.Optional[np.ndarray] = None,
        representative: str = "mean",
        classes: t.Optional[np.ndarray] = None,
        **kwargs
    ) -> t.Dict[str, t.Any]:
        """Precomputations related to cluster representative instances.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`, optional
            Instance cluster index (or target attribute).

        dist_metric : str, optional
            The distance metric used to calculate the distances between
            instances. Check :obj:`sklearn.neighbors.DistanceMetric`
            documentation for a full list of valid distance metrics.

        representative : str or :obj:`np.ndarray` or List, optional
            * If representative is string-type, then it must assume one
                value between ``median`` or ``mean``, and the selected
                method is used to estimate the representative instance of
                each class (e.g., if ``mean`` is selected, then the mean of
                attributes of all instances of the same class is used to
                represent that class).

            * If representative is a List or have :obj:`np.ndarray` type,
                then its length must be the number of different classes in
                ``y`` and each of its element must be a representative
                instance for each class. For example, the following 2-D
                array is the representative of the ``Iris`` dataset,
                calculated using the mean value of instances of the same
                class (effectively holding the same result as if the argument
                value was the character string ``mean``):

                [[ 5.006  3.428  1.462  0.246]  # 'Setosa' mean values
                 [ 5.936  2.77   4.26   1.326]  # 'Versicolor' mean values
                 [ 6.588  2.974  5.552  2.026]] # 'Virginica' mean values

                 The attribute order must be, of course, the same as the
                 original instances in the dataset.

        classes : :obj:`np.ndarray`, optional
            Distinct classes in ``y``. Used to exploit precomputations.

        **kwargs
            Additional arguments. May have previously precomputed before
            this method from other precomputed methods, so they can help
            speed up this precomputation.

        Returns
        -------
        :obj:`dict`
            The following precomputed items are returned:
                * ``pairwise_intracls_dists`` (:obj:`np.ndarray`):
                  distance between each distinct pair of instances of
                  the same class.
        """
        precomp_vals = {}

        if (
            N is not None
            and y is not None
            and not {"representative"}.issubset(kwargs)
        ):
            precomp_vals["representative"] = cls._get_class_representatives(
                N=N, y=y, representative=representative, classes=classes
            )

        return precomp_vals

    @classmethod
    def _calc_normalized_intercls_dist(
        cls,
        group_inst_a: np.ndarray,
        group_inst_b: np.ndarray,
        dist_metric: str = "euclidean",
    ) -> np.ndarray:
        """Calculate the distance between instances of different classes.

        The distance is normalized by the number of distinct pairs
        between ``group_inst_a`` and ``group_inst_b``.
        """
        norm_intercls_dist = scipy.spatial.distance.cdist(
            group_inst_a, group_inst_b, metric=dist_metric
        )

        return norm_intercls_dist / norm_intercls_dist.size

    @classmethod
    def _calc_pwise_norm_intercls_dist(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        dist_metric: str = "euclidean",
        classes: t.Optional[np.ndarray] = None,
        cls_inds: t.Optional[np.ndarray] = None,
    ) -> t.List[np.ndarray]:
        """Calculate all pairwise normalized interclass distances."""
        if cls_inds is None:
            if classes is None:
                classes = np.unique(y)

            cls_inds = _utils.calc_cls_inds(y=y, classes=classes)

        intercls_dists = [
            cls._calc_normalized_intercls_dist(
                N[cls_inds[id_cls_a, :], :],
                N[cls_inds[id_cls_b, :], :],
                dist_metric=dist_metric,
            )
            for id_cls_a, id_cls_b in itertools.combinations(
                np.arange(cls_inds.shape[0]), 2
            )
        ]

        return intercls_dists

    @classmethod
    def _calc_intracls_dists(
        cls,
        instances: np.ndarray,
        dist_metric: str = "euclidean",
        get_max_dist: bool = True,
    ) -> float:
        """Calculate the intraclass distance of the given instances.

        The intraclass is the maximum distance between two distinct
        instances of the same class. If ``get_max`` is false, then
        all distances are returned instead.
        """
        intracls_dists = scipy.spatial.distance.pdist(
            instances, metric=dist_metric
        )

        return intracls_dists.max() if get_max_dist else intracls_dists

    @classmethod
    def _calc_all_intracls_dists(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        dist_metric: str = "euclidean",
        get_max_dist: bool = True,
        cls_inds: t.Optional[np.ndarray] = None,
        classes: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate all intraclass (internal to a class) distances."""
        if cls_inds is None:
            if classes is None:
                classes = np.unique(y)

            cls_inds = _utils.calc_cls_inds(y=y, classes=classes)

        intracls_dists = np.array(
            [
                cls._calc_intracls_dists(
                    N[cur_class, :],
                    dist_metric=dist_metric,
                    get_max_dist=get_max_dist,
                )
                for cur_class in cls_inds
            ],
            dtype=object,
        )

        return intracls_dists

    @classmethod
    def _get_nearest_neighbors(
        cls,
        N: np.ndarray,
        n_neighbors: int,
        dist_metric: str = "euclidean",
    ) -> np.ndarray:
        """Indexes of ``n_neighbors`` nearest neighbors for each instance."""
        model = sklearn.neighbors.KDTree(N, metric=dist_metric)

        # Note: skip the first column because it's always the
        # instance itself
        nearest_neighbors = model.query(
            N, k=n_neighbors + 1, return_distance=False
        )[:, 1:]

        return nearest_neighbors

    @classmethod
    def _get_class_representatives(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        representative: t.Union[t.List, np.ndarray, str] = "mean",
        cls_inds: t.Optional[np.ndarray] = None,
        classes: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Get a representative instance for each distinct class.

        If ``representative`` argument is a string, then it must be
        some statistical method to be aplied in the attributes of
        instances of the same class in ``N`` to construct the class
        representative instance (currently supported only ``mean`` and
        ``median``). If ``representative`` is a sequence, then its
        shape must be (number_of_classes, number_of_attributes) (i.e.,
        there must have one class representative for each distinct class,
        and every class representative must have the same dimension of
        the instances in ``N``.)
        """
        if classes is None:
            classes = np.unique(y)

        if isinstance(representative, str):
            center_method = {"mean": np.mean, "median": np.median}.get(
                representative
            )

            if center_method is None:
                raise ValueError(
                    "'representative' must be 'mean' or "
                    "'median'. Got '{}'.".format(representative)
                )

            if cls_inds is None:
                cls_inds = _utils.calc_cls_inds(y=y, classes=classes)

            representative = [
                center_method(N[cur_class, :], axis=0)
                for cur_class in cls_inds
            ]

        elif not hasattr(representative, "__len__"):
            raise TypeError(
                "'representative' type must be string "
                "or a sequence or a numpy array. "
                "Got '{}'.".format(type(representative))
            )

        representative_arr = np.asarray(representative)

        num_repr, repr_dim = representative_arr.shape
        _, num_attr = N.shape

        if num_repr != classes.size:
            raise ValueError(
                "There must exist one class representative "
                "for every distinct class. (Expected '{}', "
                "got '{}'".format(classes.size, num_repr)
            )

        if repr_dim != num_attr:
            raise ValueError(
                "The dimension of each class representative "
                "must match the instances dimension. (Expected "
                "'{}', got '{}'".format(classes.size, repr_dim)
            )

        return representative_arr

    @classmethod
    def ft_vdu(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        dist_metric: str = "euclidean",
        cls_inds: t.Optional[np.ndarray] = None,
        classes: t.Optional[np.ndarray] = None,
        intracls_dists: t.Optional[np.ndarray] = None,
        pairwise_norm_intercls_dist: t.Optional[t.List[np.ndarray]] = None,
    ) -> float:
        """Compute the Dunn Index.

        Metric range is 0 (inclusive) and infinity.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Instance cluster index (or target attribute).

        dist_metric : str, optional
            The distance metric used to calculate the distances between
            instances. Check :obj:`scipy.spatial.distance` documentation
            for a full list of valid distance metrics. If precomputation
            in clustering metafeatures is enabled, then this parameter
            takes no effect.

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows represents each distinct class, and the columns
            represents the instances. Used to take advantage of
            precomputations.

        classes : :obj:`np.ndarray`, optional
            Distinct classes in ``y``. Used to exploit precomputations.

        intracls_dists : :obj:`np.ndarray`, optional
            Distance between the fartest pair of instances in the same
            class, for each class. Used to exploit precomputations.

        pairwise_norm_intercls_dists : :obj:`np.ndarray`, optional
            Normalized pairwise distances between instances of different
            classes.

        Returns
        -------
        float
            Dunn index for given parameters.

        References
        ----------
        .. [1] J.C. Dunn, Well-separated clusters and optimal fuzzy
           partitions, J. Cybern. 4 (1) (1974) 95–104.

        """
        if pairwise_norm_intercls_dist is None:
            pairwise_norm_intercls_dist = cls._calc_pwise_norm_intercls_dist(
                N=N,
                y=y,
                dist_metric=dist_metric,
                classes=classes,
                cls_inds=cls_inds,
            )

        if intracls_dists is None:
            intracls_dists = cls._calc_all_intracls_dists(
                N=N,
                y=y,
                dist_metric=dist_metric,
                classes=classes,
                cls_inds=cls_inds,
            )

        _min_intercls_dist = np.inf

        for vals in pairwise_norm_intercls_dist:
            _min_intercls_dist = min(_min_intercls_dist, np.min(vals))

        vdu = float(_min_intercls_dist / np.max(intracls_dists))

        return vdu

    @classmethod
    def ft_vdb(cls, N: np.ndarray, y: np.ndarray) -> float:
        """Compute the Davies and Bouldin Index.

        Metric range is 0 (inclusive) and infinity.

        Check :obj:`sklearn.metrics.davies_bouldin_score` documentation
        for more information.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Instance cluster index (or target attribute).

        References
        ----------
        .. [1] D.L. Davies, D.W. Bouldin, A cluster separation measure,
           IEEE Trans. Pattern Anal. Mach. Intell. 1 (2) (1979) 224–227.
        """
        return sklearn.metrics.davies_bouldin_score(X=N, labels=y)

    @classmethod
    def ft_int(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        dist_metric: str = "euclidean",
        cls_inds: t.Optional[np.ndarray] = None,
        classes: t.Optional[np.ndarray] = None,
        pairwise_norm_intercls_dist: t.Optional[t.List[np.ndarray]] = None,
    ) -> float:
        """Compute the INT index.

        Metric range is 0 (inclusive) and infinity.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Instance cluster index (or target attribute).

        dist_metric : str, optional
            The distance metric used to calculate the distances between
            instances. Check :obj:`scipy.spatial.distance` documentation
            for a full list of valid distance metrics. If precomputation
            in clustering metafeatures is enabled, then this parameter
            takes no effect.

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows represents each distinct class, and the columns
            represents the instances. Used to take advantage of
            precomputations.

        classes : :obj:`np.ndarray`, optional
            Distinct classes in ``y``. Used to exploit precomputations.

        pairwise_norm_intercls_dists : :obj:`np.ndarray`, optional
            Normalized pairwise distances between instances of different
            classes. Used to exploit precomputations.

        Returns
        -------
        float
            INT index.

        References
        ----------
        .. [1] Bezdek, J. C.; Pal, N. R. (1998a). Some new indexes of
           cluster validity. IEEE Transactions on Systems, Man, and
           Cybernetics, Part B, v.28, n.3, p.301–315.

        """
        if classes is None:
            classes = np.unique(y)

        class_num = classes.size

        if class_num == 1:
            return np.nan

        if pairwise_norm_intercls_dist is None:
            pairwise_norm_intercls_dist = cls._calc_pwise_norm_intercls_dist(
                N=N,
                y=y,
                dist_metric=dist_metric,
                classes=classes,
                cls_inds=cls_inds,
            )

        norm_factor = 2.0 / (class_num * (class_num - 1.0))

        _sum_intercls_dist = 0.0

        for vals in pairwise_norm_intercls_dist:
            _sum_intercls_dist += float(np.sum(vals))

        return _sum_intercls_dist * norm_factor

    @classmethod
    def ft_sil(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        dist_metric: str = "euclidean",
        sample_frac: t.Optional[int] = None,
        random_state: t.Optional[int] = None,
    ) -> float:
        """Compute the mean silhouette value.

        Metric range is -1 to +1 (both inclusive).

        Check :obj:`sklearn.metrics.silhouette_score` documentation for
        more information.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Instance cluster index (or target attribute).

        dist_metric : str, optional
            The distance metric used to calculate the distances between
            instances. Check :obj:`sklearn.neighbors.DistanceMetric`
            documentation for a full list of valid distance metrics.

        sample_frac : int, optional
            Sample fraction used to compute the silhouette coefficient. If
            None is given, then all data is used.

        random_state : int, optional
            Used if ``sample_frac`` is not None. Random seed used while
            sampling the data.

        Returns
        -------
        float
            Mean Silhouette value.

        References
        ----------
        .. [1] P.J. Rousseeuw, Silhouettes: a graphical aid to the
           interpretation and validation of cluster analysis, J.
           Comput. Appl. Math. 20 (1987) 53–65.
        """
        sample_size = N.shape[0]

        if sample_frac is not None:
            sample_size = int(sample_frac * sample_size)

        silhouette = sklearn.metrics.silhouette_score(
            X=N,
            labels=y,
            metric=dist_metric,
            sample_size=sample_size,
            random_state=random_state,
        )

        return silhouette

    @classmethod
    def ft_pb(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        dist_metric: str = "euclidean",
    ) -> float:
        """Compute the pearson correlation between class matching and instance
        distances.

        The measure interval is -1 and +1 (inclusive).

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Instance cluster index (or target attribute).

        dist_metric : str, optional
            The distance metric used to calculate the distances between
            instances. Check :obj:`scipy.spatial.distance` for a full
            list of valid distance metrics.

        Returns
        -------
        float
            Point Biserial coefficient.

        References
        ----------
        .. [1] J. Lev, "The Point Biserial Coefficient of Correlation", Ann.
           Math. Statist., Vol. 20, no.1, pp. 125-126, 1949.

        """
        inst_dists = scipy.spatial.distance.pdist(X=N, metric=dist_metric)

        inst_matching_classes = np.array(
            [
                inst_class_a == inst_class_b
                for inst_class_a, inst_class_b in itertools.combinations(y, 2)
            ]
        )

        correlation, _ = scipy.stats.pointbiserialr(
            x=inst_matching_classes, y=inst_dists
        )

        return correlation

    @classmethod
    def ft_ch(cls, N: np.ndarray, y: np.ndarray) -> float:
        """Compute the Calinski and Harabasz index.

        Check :obj:`sklearn.metrics.calinski_harabasz_score` documentation
        for more information.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Instance cluster index (or target attribute).

        Returns
        -------
        float
            Calinski-Harabasz index.

        References
        ----------
        .. [1] T. Calinski, J. Harabasz, A dendrite method for cluster
           analysis, Commun. Stat. Theory Methods 3 (1) (1974) 1–27.
        """
        return sklearn.metrics.calinski_harabasz_score(X=N, labels=y)

    @classmethod
    def ft_nre(
        cls,
        y: np.ndarray,
        class_freqs: t.Optional[np.ndarray] = None,
    ) -> float:
        """Compute the normalized relative entropy.

        An indicator of uniformity distributed of instances among clusters.

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Instance cluster index (or target attribute).

        class_freqs : :obj:`np.ndarray`, optional
            Absolute class frequencies. Used to exploit precomputations.

        Returns
        -------
        float
            Entropy of relative class frequencies.

        References
        ----------
        .. [1] Bruno Almeida Pimentel, André C.P.L.F. de Carvalho.
           A new data characterization for selecting clustering algorithms
           using meta-learning. Information Sciences, Volume 477, 2019,
           Pages 203-219.
        """
        if class_freqs is None:
            _, class_freqs = np.unique(y, return_counts=True)

        num_inst = y.size

        return scipy.stats.entropy(class_freqs / num_inst)

    @classmethod
    def ft_sc(
        cls,
        y: np.ndarray,
        size: int = 15,
        normalize: bool = False,
        class_freqs: t.Optional[np.ndarray] = None,
    ) -> int:
        """Compute the number of clusters with size smaller than a given size.

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Instance cluster index (or target attribute).

        size : int, optional
            Maximum (exclusive) size of classes to be considered.

        normalize : bool, optional
            If True, then the result will be the proportion of classes
            with less than ``size`` instances from the total of classes.
            (i.e., result is divided by the number of classes.)

        class_freqs : :obj:`np.ndarray`, optional
            Class (absolute) frequencies. Used to exploit precomputations.

        Returns
        -------
        int or float
            Number of classes with less than ``size`` instances if
            ``normalize`` is False, proportion of classes with less
            than ``size`` instances otherwise.

        References
        ----------
        .. [1] Bruno Almeida Pimentel, André C.P.L.F. de Carvalho.
           A new data characterization for selecting clustering algorithms
           using meta-learning. Information Sciences, Volume 477, 2019,
           Pages 203-219.
        """
        if class_freqs is None:
            _, class_freqs = np.unique(y, return_counts=True)

        quant = (class_freqs < size).sum()

        if normalize:
            quant /= class_freqs.size

        return quant
