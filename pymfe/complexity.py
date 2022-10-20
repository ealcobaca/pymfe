"""Module dedicated to extraction of complexity metafeatures."""

import typing as t
import itertools

import numpy as np
import sklearn
import sklearn.pipeline
import scipy.spatial
import igraph
import gower

from pymfe.general import MFEGeneral
from pymfe.clustering import MFEClustering
from pymfe import _utils


class MFEComplexity:
    """Keep methods for metafeatures of ``Complexity`` group.

    The convention adopted for metafeature extraction related methods is to
    always start with ``ft_`` prefix to allow automatic method detection. This
    prefix is predefined within ``_internal`` module.

    All method signature follows the conventions and restrictions listed below:

    1. For independent attribute data, ``X`` means ``every type of attribute``,
       ``N`` means ``Numeric attributes only`` and ``C`` stands for
       ``Categorical attributes only``. It is important to note that the
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
       value or a generic List (preferably a :obj:`np.ndarray`) type with
       numeric values.

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
    def precompute_complexity(
        cls, y: t.Optional[np.ndarray] = None, **kwargs
    ) -> t.Dict[str, t.Any]:
        """Precompute some useful things to support feature-based measures.

        Parameters
        ----------
        y : :obj:`np.ndarray`, optional
            Target attribute.

        **kwargs
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
        :obj:`dict`
            With following precomputed items:
                - ``ovo_comb`` (list): List of all class OVO combination,
                    i.e., all combinations of distinct class indices by pairs
                    ([(0, 1), (0, 2) ...].)
                - ``cls_inds`` (:obj:`np.ndarray`): Boolean array which
                    indicates whether each example belongs to each class. The
                    rows corresponds to the distinct classes, and the instances
                    are represented by the columns.
                - ``classes`` (:obj:`np.ndarray`): distinct classes in the
                    fitted target attribute.
                - ``class_freqs`` (:obj:`np.ndarray`): The number of examples
                    in each class. The indices corresponds to the classes.
        """
        precomp_vals = {}  # type: t.Dict[str, t.Any]

        if y is not None and not {"classes", "class_freqs"}.issubset(kwargs):
            sub_dic = MFEGeneral.precompute_general_class(y)
            precomp_vals.update(sub_dic)

        classes = kwargs.get("classes", precomp_vals.get("classes"))

        if y is not None and "cls_inds" not in kwargs:
            cls_inds = _utils.calc_cls_inds(y, classes)
            precomp_vals["cls_inds"] = cls_inds

        if y is not None and "ovo_comb" not in kwargs:
            ovo_comb = cls._calc_ovo_comb(classes)
            precomp_vals["ovo_comb"] = ovo_comb

        return precomp_vals

    @classmethod
    def precompute_pca_tx(
        cls,
        N: np.ndarray,
        tx_n_components: float = 0.95,
        random_state: t.Optional[int] = None,
        **kwargs
    ) -> t.Dict[str, int]:
        """Precompute PCA to support dimensionality measures.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        tx_n_components : float, optional
            Specifies the number of components such that the amount of variance
            that needs to be explained is greater than the percentage specified
            by ``tx_n_components``. The PCA is computed using ``N``.

        random_state : int, optional
            If the fitted data is huge and the number of principal components
            to be kept is low, then the PCA analysis is done using a randomized
            strategy for efficiency. This random seed keeps the results
            replicable. Check ``sklearn.decomposition.PCA`` documentation for
            more information.

        **kwargs
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
        :obj:`dict`
            With following precomputed items:
                - ``num_attr_pca`` (int): Number of features after PCA
                    analysis with at least ``tx_n_components`` fraction of
                    data variance explained by the selected principal
                    components.
        """
        precomp_vals = {}  # type: t.Dict[str, t.Any]

        if N is not None and "num_attr_pca" not in kwargs:
            pca = sklearn.decomposition.PCA(
                n_components=tx_n_components, random_state=random_state
            )

            pca.fit(N)

            num_attr_pca = pca.explained_variance_ratio_.shape[0]

            precomp_vals["num_attr_pca"] = num_attr_pca

        return precomp_vals

    @classmethod
    def precompute_complexity_svm(
        cls,
        y: t.Optional[np.ndarray] = None,
        max_iter: t.Union[int, float] = 1e5,
        random_state: t.Optional[int] = None,
        **kwargs
    ) -> t.Dict[str, sklearn.pipeline.Pipeline]:
        """Init a Support Vector Classifier pipeline (with data standardization.)

        Parameters
        ----------
        max_iter : float or int, optional
            Maximum number of iterations allowed for the support vector
            machine model convergence. This parameter can receive float
            numbers to be compatible with the Python scientific notation
            data type.

        random_state : int, optional
            Random seed for dual coordinate descent while fitting the
            Support Vector Classifier model. Check `sklearn.svm.LinearSVC`
            documentation (`random_state` parameter) for more information.

        **kwargs
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
        :obj:`dict`
            With following precomputed items:
                - ``svc_pipeline`` (sklearn.pipeline.Pipeline): support
                    vector classifier learning pipeline, with data
                    standardization (mean = 0 and variance = 1) before the
                    learning model.
        """
        precomp_vals = {}  # type: t.Dict[str, t.Any]

        if y is not None and "svc_pipeline" not in kwargs:
            scaler = sklearn.preprocessing.StandardScaler()

            # Note: 'C' parameter is inversely proportional to the
            # regularization strenght, which is '0.5' in the reference
            # paper.
            svc = sklearn.svm.LinearSVC(
                penalty="l2",
                loss="hinge",
                C=2.0,
                tol=10e-3,
                max_iter=int(max_iter),
                random_state=random_state,
            )

            pip = sklearn.pipeline.Pipeline([("scaler", scaler), ("svc", svc)])

            precomp_vals["svc_pipeline"] = pip

        return precomp_vals

    @classmethod
    def precompute_norm_dist_mat(
        cls, N: np.ndarray, metric: str = "gower", p: t.Union[int, float] = 2, **kwargs
    ) -> t.Dict[str, np.ndarray]:
        """Precompute normalized ``N`` and pairwise distance among instances.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        metric : str, optional
            Metric used to calculate the distances between the instances.
            Check the ``scipy.spatial.distance.cdist`` documentation to
            get a list of all available metrics.

        p : int, optional
            Power parameter for the Minkowski metric. When p = 1, this is
            equivalent to using Manhattan distance (l1), and Euclidean
            distance (l2) for p = 2. For arbitrary p, Minkowski distance
            (l_p) is used.

        **kwargs
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
        :obj:`dict`
            With following precomputed items:
                - ``N_scaled`` (:obj:`np.ndarray`): numerical data ``N`` with
                    each feature normalized  in [0, 1] range. Used only if
                    ``norm_dist_mat`` is None.
                - ``norm_dist_mat`` (:obj:`np.ndarray`): square matrix with
                    the normalized pairwise distances between each instance
                    in ``N_scaled``, i.e., between the normalized instances.
                    (note that this matrix is the normalized pairwise
                    distances between the normalized instances, i.e., there
                    is two normalization processes involved.)
                - ``orig_dist_mat_min`` (float): minimal value from the
                    original pairwise distance matrix. Can be used to
                    preprocess test data before predictions.
                - ``orig_dist_mat_min`` (float): range (max - min) value from
                    the original pairwise distance matrix. Can be used to
                    preprocess test data before predictions.
        """
        precomp_vals = {}  # type: t.Dict[str, t.Any]

        N_scaled = kwargs.get("N_scaled")

        if N.size and N_scaled is None:
            N_scaled = cls._scale_N(N=N)
            precomp_vals["N_scaled"] = N_scaled

        _all_precomputed = {
            "norm_dist_mat",
            "orig_dist_mat_min",
            "orig_dist_mat_ptp",
        }.issubset(kwargs)

        if N_scaled is not None and not _all_precomputed:
            (
                norm_dist_mat,
                orig_dist_mat_min,
                orig_dist_mat_ptp,
            ) = cls._calc_norm_dist_mat(N=N, metric=metric, p=p, N_scaled=N_scaled)

            precomp_vals["norm_dist_mat"] = norm_dist_mat
            precomp_vals["orig_dist_mat_min"] = orig_dist_mat_min
            precomp_vals["orig_dist_mat_ptp"] = orig_dist_mat_ptp

        return precomp_vals

    @classmethod
    def precompute_nearest_enemy(
        cls,
        N: np.ndarray,
        y: t.Optional[np.ndarray] = None,
        metric: str = "gower",
        p: t.Union[int, float] = 2,
        **kwargs
    ) -> t.Dict[str, np.ndarray]:
        """Precompute instances nearest enemy related values.

        The instance nearest enemy is the nearest instance from a
        different class.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        metric : str, optional
            Metric used to calculate the distances between the instances.
            Check the ``scipy.spatial.distance.cdist`` documentation to
            get a list of all available metrics.

        p : int, optional
            Power parameter for the Minkowski metric. When p = 1, this is
            equivalent to using Manhattan distance (l1), and Euclidean
            distance (l2) for p = 2. For arbitrary p, Minkowski distance
            (l_p) is used.

        **kwargs
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
        :obj:`dict`
            With following precomputed items:
                - ``nearest_enemy_dist`` (:obj:`np.ndarray`): distance of each
                    instance to its nearest enemy (instances of a distinct
                    class.)
                - ``nearest_enemy_ind`` (:obj:`np.ndarray`): index of the
                    nearest enemy (instances of a distinct class) for each
                    instance.

            This precomputation method also depends on values precomputed in
            other precomputation methods, ``precompute_complexity`` and
            ``precompute_norm_dist_mat``. Therefore, the return values of
            those methods can also be returned in case they are not called
            before. Check the documentation of each method to verify which
            values either methods returns to a precise description for the
            additional values that may be returned by this method.
        """
        precomp_vals = {}

        if (
            y is not None
            and N.size
            and not {"nearest_enemy_dist", "nearest_enemy_ind"}.issubset(kwargs)
        ):
            norm_dist_mat = kwargs.get("norm_dist_mat")
            cls_inds = kwargs.get("cls_inds")

            if norm_dist_mat is None:
                precomp_vals.update(
                    cls.precompute_norm_dist_mat(N=N, metric=metric, p=p, **kwargs)
                )
                norm_dist_mat = precomp_vals["norm_dist_mat"]

            if cls_inds is None:
                precomp_vals.update(cls.precompute_complexity(y=y, **kwargs))
                cls_inds = precomp_vals["cls_inds"]

            nearest_enemy_dist, nearest_enemy_ind = cls._calc_nearest_enemies(
                norm_dist_mat=norm_dist_mat,
                cls_inds=cls_inds,
            )

            precomp_vals["nearest_enemy_dist"] = nearest_enemy_dist
            precomp_vals["nearest_enemy_ind"] = nearest_enemy_ind

        return precomp_vals

    @classmethod
    def precompute_adjacency_graph(
        cls,
        N: np.ndarray,
        y: t.Optional[np.ndarray] = None,
        metric: str = "gower",
        p: float = 2.0,
        n_jobs: t.Optional[int] = None,
        **kwargs
    ) -> t.Dict[str, np.ndarray]:
        """Precompute instances nearest enemy related values.

        The instance nearest enemy is the nearest instance from a
        different class.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        **kwargs
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
        :obj:`dict`
            With following precomputed items:
        """
        precomp_vals = {}

        if y is not None and N.size and "adj_graph" not in kwargs:
            norm_dist_mat = kwargs.get("norm_dist_mat")
            N_scaled = kwargs.get("N_scaled")
            cls_inds = kwargs.get("cls_inds")

            try:
                adj_graph = cls._build_adjacency_graph(
                    N=N,
                    y=y,
                    cls_inds=cls_inds,
                    n_jobs=n_jobs,
                    metric=metric,
                    norm_dist_mat=norm_dist_mat,
                    N_scaled=N_scaled,
                    p=p,
                )

            except Exception as err:
                raise Exception from err

            precomp_vals["adj_graph"] = adj_graph

        return precomp_vals

    @classmethod
    def _calc_norm_dist_mat(
        cls,
        N: np.ndarray,
        metric: str,
        p: t.Union[int, float] = 2,
        N_scaled: t.Optional[np.ndarray] = None,
        normalize: bool = True,
    ) -> t.Tuple[np.ndarray, float, float]:
        """Calculate a pairwise normalized distance matrix.

        All distances are normalized in [0, 1] range.

        If ``normalize`` is False, then the unnormalized pairwise
        distance matrix is returned.

        If ``return_scalers`` is True, this method also returns
        the global minimal distance `Min`, and the range (max - min)
        `Ptp` (point-to-point) values, used to normalize the distance
        matrix. Hence, the original distance matrix M can be obtained
        using the following relationship: M = M_scaled * Ptp + Min.

        The values `Ptp` and `Min` can also be used to normalize
        test data, in order to provide nearest neighbor predictions
        without data leakage.
        """
        if N_scaled is None:
            N_scaled = cls._scale_N(N=N)

        if metric == "gower":
            norm_dist_mat = gower.gower_matrix(N_scaled)
            return norm_dist_mat, 0.0, 1.0

        norm_dist_mat = scipy.spatial.distance.cdist(
            N_scaled,
            N_scaled,
            metric=metric,
            p=p,
        )

        orig_dist_mat_min = float(np.min(norm_dist_mat))
        orig_dist_mat_ptp = float(np.ptp(norm_dist_mat))

        if normalize and np.not_equal(0.0, orig_dist_mat_ptp):
            norm_dist_mat = (norm_dist_mat - orig_dist_mat_min) / orig_dist_mat_ptp

        return norm_dist_mat, orig_dist_mat_min, orig_dist_mat_ptp

    @classmethod
    def _build_adjacency_graph(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        metric: str = "gower",
        p: float = 2.0,
        radius_frac: float = 0.15,
        n_jobs: t.Optional[int] = None,
        cls_inds: t.Optional[np.ndarray] = None,
        N_scaled: t.Optional[np.ndarray] = None,
        norm_dist_mat: t.Optional[np.ndarray] = None,
    ):
        """Build a adjacency graph for nearest neighbors of a same class."""
        if cls_inds is None:
            cls_inds = _utils.calc_cls_inds(y)

        if norm_dist_mat is None:
            norm_dist_mat, _, _ = cls._calc_norm_dist_mat(
                N=N,
                metric=metric,
                p=p,
                N_scaled=N_scaled,
            )

        num_inst, _ = N.shape

        n_neighbors = int(
            round(num_inst * radius_frac) if 0 < radius_frac < 1.0 else radius_frac
        )
        n_neighbors = max(n_neighbors, 1)

        adj_mat = sklearn.neighbors.kneighbors_graph(
            norm_dist_mat,
            n_neighbors=n_neighbors,
            mode="distance",
            include_self=False,
            n_jobs=n_jobs,
            metric="precomputed",
        ).toarray()

        for inds_cur_cls in cls_inds:
            # Note: filtering out neighbors of distinct classes.
            # 'inds_cur_cls' is a boolean array of shape (num_inst,), where:
            #  inds_cur_cls[i] = 1: i-th instance belongs to the current class;
            #  inds_cur_cls[i] = 0: otherwise.
            adj_mat[inds_cur_cls] *= inds_cur_cls.astype(float)

        # Note: element-wise maximum to turn 'adj_mat' symmetric
        np.maximum(adj_mat, adj_mat.T, out=adj_mat)
        adj_graph = igraph.Graph.Weighted_Adjacency(adj_mat, mode="undirected")

        return adj_graph

    @staticmethod
    def _calc_ovo_comb(classes: np.ndarray) -> np.ndarray:
        """Compute the ``ovo_comb`` variable.

        The ``ovo_comb`` value is a array with all class OVO combination,
        i.e., all combinations of distinct class indices by pairs.
        """
        ovo_comb = itertools.combinations(np.arange(classes.size), 2)
        return np.asarray(list(ovo_comb), dtype=int)

    @staticmethod
    def _calc_maxmax(N_cls_1: np.ndarray, N_cls_2: np.ndarray) -> np.ndarray:
        """Compute the maximum of the maximum values per class for all feat.

        The index i indicate the maxmax of feature i.
        """
        max_cls_1 = np.max(N_cls_1, axis=0) if N_cls_1.size else -np.inf
        max_cls_2 = np.max(N_cls_2, axis=0) if N_cls_2.size else -np.inf

        maxmax = np.maximum(max_cls_1, max_cls_2)

        if not maxmax.shape:
            return np.full(shape=N_cls_1.shape[1], fill_value=-np.inf)

        return maxmax

    @staticmethod
    def _calc_minmin(N_cls_1: np.ndarray, N_cls_2: np.ndarray) -> np.ndarray:
        """Compute the minimum of the minimum values per class for all feat.

        The index i indicate the minmin of feature i.
        """
        min_cls_1 = np.min(N_cls_1, axis=0) if N_cls_1.size else np.inf
        min_cls_2 = np.min(N_cls_2, axis=0) if N_cls_2.size else np.inf

        minmin = np.minimum(min_cls_1, min_cls_2)

        if not minmin.shape:
            return np.full(shape=N_cls_1.shape[1], fill_value=np.inf)

        return minmin

    @staticmethod
    def _calc_minmax(N_cls_1: np.ndarray, N_cls_2: np.ndarray) -> np.ndarray:
        """Compute the minimum of the maximum values per class for all feat.

        The index i indicate the minmax of feature i.
        """
        if N_cls_1.size == 0 or N_cls_2.size == 0:
            # Note: if there is no two classes, the 'overlapping region'
            # becomes ill defined. Thus, returning '-np.inf' alongside
            # '_calc_maxmin()' returning '+np.inf' guarantees that no
            # example will remain into the (undefined) 'overlapping region.'
            return np.full(shape=N_cls_1.shape[1], fill_value=-np.inf)

        minmax = np.minimum(np.max(N_cls_1, axis=0), np.max(N_cls_2, axis=0))

        return minmax

    @staticmethod
    def _calc_maxmin(N_cls_1: np.ndarray, N_cls_2: np.ndarray) -> np.ndarray:
        """Compute the maximum of the minimum values per class for all feat.

        The index i indicate the maxmin of the ith feature.
        """
        if N_cls_1.size == 0 or N_cls_2.size == 0:
            # Note: if there is no two classes, the 'overlapping region'
            # becomes ill defined. Thus, returning '+np.inf' alongside
            # '_calc_minmax()' returning '-np.inf' guarantees that no
            # example will remain into the (undefined) 'overlapping region.'
            return np.full(shape=N_cls_1.shape[1], fill_value=np.inf)

        maxmin = np.maximum(np.min(N_cls_1, axis=0), np.min(N_cls_2, axis=0))

        return maxmin

    @staticmethod
    def _calc_overlap(
        N: np.ndarray,
        minmax: np.ndarray,
        maxmin: np.ndarray,
    ) -> t.Tuple[int, np.ndarray, np.ndarray]:
        """Compute the instances in overlapping region by feature."""
        # True if the example is in the overlapping region
        feat_overlapped_region = np.logical_and(N >= maxmin, N <= minmax)

        feat_overlap_num = np.sum(feat_overlapped_region, axis=0)
        ind_less_overlap = int(np.argmin(feat_overlap_num))

        feat_overlapped_region = np.asarray(feat_overlapped_region, dtype=bool)
        feat_overlap_num = np.asarray(feat_overlap_num, dtype=int)

        return ind_less_overlap, feat_overlap_num, feat_overlapped_region

    @classmethod
    def _interpolate(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        cls_inds: np.ndarray,
        random_state: t.Optional[int] = None,
    ) -> t.Tuple[np.ndarray, np.ndarray]:
        """Create a new dataset using interpolated instances from ``N``.

        The instances are interpolated using only other instances
        from the same class.
        """
        if random_state is not None:
            np.random.seed(random_state)

        N_interpol = np.atleast_2d(np.zeros(N.shape, dtype=N.dtype))
        y_interpol = np.zeros(y.shape, dtype=y.dtype)

        ind_cur = 0

        for inds_cur_cls in cls_inds:
            N_cur_cls = N[inds_cur_cls, :]
            subset_size = N_cur_cls.shape[0]

            # Currently it is allowed to a instance 'interpolate with itself',
            # which holds the instance itself as result.
            sample_a = N_cur_cls[np.random.choice(subset_size, subset_size), :]
            sample_b = N_cur_cls[np.random.choice(subset_size, subset_size), :]

            rand_delta = np.random.ranf(N_cur_cls.shape)

            N_subset_interp = sample_a + (sample_b - sample_a) * rand_delta

            ind_next = ind_cur + subset_size
            N_interpol[ind_cur:ind_next, :] = N_subset_interp
            y_interpol[ind_cur:ind_next] = y[inds_cur_cls]
            ind_cur = ind_next

        return N_interpol, y_interpol

    @classmethod
    def _calc_nearest_enemies(
        cls,
        norm_dist_mat: np.ndarray,
        cls_inds: np.ndarray,
    ) -> t.Tuple[np.ndarray, np.ndarray]:
        """Calculate each instances nearest enemies.

        Returns the nearest enemies distance and its indices.
        """
        num_inst = norm_dist_mat.shape[0]

        # Note: 'n_en' stands for 'nearest_enemy'
        n_en_dist = np.full(num_inst, fill_value=np.inf, dtype=float)

        n_en_inds = np.full(num_inst, fill_value=-1, dtype=int)

        for inds_cur_cls in cls_inds:
            norm_dist_en = norm_dist_mat[~inds_cur_cls, :][:, inds_cur_cls]

            en_inds = np.argmin(norm_dist_en, axis=0)
            _aux = np.arange(norm_dist_en.shape[1])

            n_en_inds[inds_cur_cls] = en_inds
            n_en_dist[inds_cur_cls] = norm_dist_en[en_inds, _aux]

        return n_en_dist, n_en_inds

    @staticmethod
    def _scale_N(N: np.ndarray) -> np.ndarray:
        """Scale all features of N to [0, 1] range."""
        N_scaled = N

        if not np.allclose(1.0, np.max(N, axis=0)) or not np.allclose(
            0.0, np.min(N, axis=0)
        ):
            N_scaled = sklearn.preprocessing.MinMaxScaler(
                feature_range=(0, 1)
            ).fit_transform(N)

        return np.atleast_2d(N_scaled)

    @classmethod
    def ft_f1(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        cls_inds: t.Optional[np.ndarray] = None,
        class_freqs: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Maximum Fisher's discriminant ratio.

        It measures theoverlap between the values of the features in
        different classes.

        The average value of this measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows corresponds to each distinct class, and the columns
            corresponds to the instances.

        class_freqs : :obj:`np.ndarray`, optional
            The number of examples in each class. The indices corresponds to
            the classes.

        Returns
        -------
        :obj:`np.ndarray`
            Inverse of all Fisher's discriminant ratios.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           (Cited on page 9). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        .. [2] Ramón A Mollineda, José S Sánchez, and José M Sotoca. Data
           characterization for effective prototype selection. In 2nd Iberian
           Conference on Pattern Recognition and Image Analysis (IbPRIA),
           pages 27–34, 2005.
        """
        classes = None

        if class_freqs is None:
            classes, class_freqs = np.unique(y, return_counts=True)

        if cls_inds is None:
            cls_inds = _utils.calc_cls_inds(y, classes)

        mean_global = np.mean(N, axis=0)

        centroids = np.asarray(
            [np.mean(N[inds_cur_cls, :], axis=0) for inds_cur_cls in cls_inds],
            dtype=float,
        )

        _numer = np.sum(np.square(centroids - mean_global).T * class_freqs, axis=1)

        _denom = np.sum(
            [
                sum(np.square(N[inds_cur_cls, :] - centroids[cls_ind, :]))
                for cls_ind, inds_cur_cls in enumerate(cls_inds)
            ],
            axis=0,
        )

        attr_discriminant_ratio = _numer / _denom

        # Note: in the reference paper, this measure is calculated as:
        # f1 = 1.0 / (1.0 + np.max(attr_discriminant_ratio))
        # But in the R package 'ECoL', to enable summarization, it is
        # calculated as:
        f1 = 1.0 / (1.0 + attr_discriminant_ratio)

        return f1

    @classmethod
    def ft_f1v(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        ovo_comb: t.Optional[np.ndarray] = None,
        cls_inds: t.Optional[np.ndarray] = None,
        class_freqs: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Directional-vector maximum Fisher's discriminant ratio.

        This measure searches for a vector which can separate the two
        classes after the examples have been projected into it and
        considers a directional Fisher criterion. Check the references
        for more information.

        The average value of this measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        ovo_comb : :obj:`np.ndarray`, optional
            List of all class OVO combination, i.e., all combinations of
            distinct class indices by pairs ([(0, 1), (0, 2) ...].)

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows corresponds to each distinct class, and the columns
            corresponds to the instances.

        class_freqs : :obj:`np.ndarray`, optional
            The number of examples in each class. The indices corresponds to
            the classes.

        Returns
        -------
        :obj:`np.ndarray`
            Inverse of directional vector of Fisher's discriminant ratio.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           (Cited on page 9). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        .. [2] Witold Malina. Two-parameter fisher criterion. IEEE
           Transactions on Systems, Man, and Cybernetics, Part B
           (Cybernetics), 31(4):629–636, 2001.
        """
        if ovo_comb is None or cls_inds is None or class_freqs is None:
            sub_dic = cls.precompute_complexity(y=y)
            ovo_comb = sub_dic["ovo_comb"]
            cls_inds = sub_dic["cls_inds"]
            class_freqs = sub_dic["class_freqs"]

        _ovo_comb = np.asarray(ovo_comb, dtype=int)
        _class_freqs = np.asarray(class_freqs, dtype=int)
        _cls_inds = np.asarray(cls_inds, dtype=bool)

        num_attr = N.shape[1]

        df = np.zeros(_ovo_comb.shape[0], dtype=float)
        mat_scatter_within = []
        centroids = np.zeros((_class_freqs.size, num_attr), dtype=float)

        for cls_ind, inds_cur_cls in enumerate(_cls_inds):
            cur_cls_inst = N[inds_cur_cls, :]
            mat_scatter_within.append(np.cov(cur_cls_inst, rowvar=False, ddof=1))
            centroids[cls_ind, :] = cur_cls_inst.mean(axis=0)

        for ind, (cls_id_1, cls_id_2) in enumerate(_ovo_comb):
            centroid_diff = (centroids[cls_id_1, :] - centroids[cls_id_2, :]).reshape(
                -1, 1
            )

            total_inst_num = _class_freqs[cls_id_1] + _class_freqs[cls_id_2]

            W_mat = (
                _class_freqs[cls_id_1] * mat_scatter_within[cls_id_1]
                + _class_freqs[cls_id_2] * mat_scatter_within[cls_id_2]
            ) / total_inst_num

            # Note: the result of np.linalg.piv 'Moore-Penrose' pseudo-inverse
            # does not match with the result of MASS::ginv 'Moore-Penrose'
            # pseudo-inverse implementation. The metafeature final result does
            # not seems to be affected.
            direc = np.matmul(scipy.linalg.pinv(W_mat), centroid_diff)
            mat_scatter_between = np.outer(centroid_diff, centroid_diff)

            _numen = np.matmul(direc.T, np.matmul(mat_scatter_between, direc))
            _denom = np.matmul(direc.T, np.matmul(W_mat, direc))

            df[ind] = _numen / _denom

        f1v = 1.0 / (1.0 + df)

        return f1v

    @classmethod
    def ft_f2(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        ovo_comb: t.Optional[np.ndarray] = None,
        cls_inds: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Volume of the overlapping region.

        This measure calculates the overlap of the distributions of
        the features values within the classes.

        This measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Fitted target attribute.

        ovo_comb : :obj:`np.ndarray`, optional
            List of all class OVO combination, i.e., all combinations of
            distinct class indices by pairs ([(0, 1), (0, 2) ...].)

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows corresponds to each distinct class, and the columns
            corresponds to the instances.

        Returns
        -------
        :obj:`np.ndarray`
            Volume of the overlapping region for each OVO combination.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           (Cited on page 9). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        .. [2] Marcilio C P Souto, Ana C Lorena, Newton Spolaôr, and Ivan G
           Costa. Complexity measures of supervised classification tasks: a
           case study for cancer gene expression data. In International
           Joint Conference on Neural Networks (IJCNN), pages 1352–1358,
           2010.
        .. [3] Lisa Cummins. Combining and Choosing Case Base Maintenance
           Algorithms. PhD thesis, National University of Ireland, Cork,
           2013.
        """
        if ovo_comb is None or cls_inds is None:
            sub_dic = cls.precompute_complexity(y=y)
            ovo_comb = sub_dic["ovo_comb"]
            cls_inds = sub_dic["cls_inds"]

        _ovo_comb = np.asarray(ovo_comb, dtype=int)
        _cls_inds = np.asarray(cls_inds, dtype=bool)

        f2 = np.zeros(_ovo_comb.shape[0], dtype=float)

        for ind, (cls_id_1, cls_id_2) in enumerate(_ovo_comb):
            N_cls_1 = N[_cls_inds[cls_id_1], :]
            N_cls_2 = N[_cls_inds[cls_id_2], :]

            maxmax = cls._calc_maxmax(N_cls_1, N_cls_2)
            minmin = cls._calc_minmin(N_cls_1, N_cls_2)
            minmax = cls._calc_minmax(N_cls_1, N_cls_2)
            maxmin = cls._calc_maxmin(N_cls_1, N_cls_2)

            f2[ind] = np.prod(np.maximum(0.0, minmax - maxmin) / (maxmax - minmin))

        return f2

    @classmethod
    def ft_f3(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        ovo_comb: t.Optional[np.ndarray] = None,
        cls_inds: t.Optional[np.ndarray] = None,
        class_freqs: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute feature maximum individual efficiency.

        The average value of this measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        ovo_comb : :obj:`np.ndarray`, optional
            List of all class OVO combination, i.e., all combinations of
            distinct class indices by pairs ([(0, 1), (0, 2) ...].)

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows corresponds to each distinct class, and the columns
            corresponds to the instances.

        class_freqs : :obj:`np.ndarray`, optional
            The number of examples in each class. The indices corresponds to
            the classes.

        Returns
        -------
        :obj:`np.ndarray`
            An array with the maximum individual feature efficiency measure for
            each feature.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           (Cited on page 6). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        """
        if ovo_comb is None or cls_inds is None or class_freqs is None:
            sub_dic = cls.precompute_complexity(y=y)
            ovo_comb = sub_dic["ovo_comb"]
            cls_inds = sub_dic["cls_inds"]
            class_freqs = sub_dic["class_freqs"]

        _ovo_comb = np.asarray(ovo_comb, dtype=int)
        _class_freqs = np.asarray(class_freqs, dtype=int)
        _cls_inds = np.asarray(cls_inds, dtype=bool)

        f3 = np.zeros(_ovo_comb.shape[0], dtype=float)

        for ind, (cls_id_1, cls_id_2) in enumerate(_ovo_comb):
            N_cls_1 = N[_cls_inds[cls_id_1, :], :]
            N_cls_2 = N[_cls_inds[cls_id_2, :], :]

            ind_less_overlap, feat_overlap_num, _ = cls._calc_overlap(
                N=N,
                minmax=cls._calc_minmax(N_cls_1, N_cls_2),
                maxmin=cls._calc_maxmin(N_cls_1, N_cls_2),
            )

            f3[ind] = feat_overlap_num[ind_less_overlap] / (
                _class_freqs[cls_id_1] + _class_freqs[cls_id_2]
            )

        # The measure is computed in the literature using the mean. However, it
        # is formulated here as a meta-feature. Therefore, the post-processing
        # should be used to get the mean and other measures as well.
        return f3

    @classmethod
    def ft_f4(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        ovo_comb: t.Optional[np.ndarray] = None,
        cls_inds: t.Optional[np.ndarray] = None,
        class_freqs: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the collective feature efficiency.

        The average value of this measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        ovo_comb : :obj:`np.ndarray`, optional
            List of all class OVO combination, i.e., all combinations of
            distinct class indices by pairs ([(0, 1), (0, 2) ...].)

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows corresponds to each distinct class, and the columns
            corresponds to the instances.

        class_freqs : :obj:`np.ndarray`, optional
            The number of examples in each class. The indices corresponds to
            the classes.

        Returns
        -------
        :obj:`np.ndarray`
            An array with the collective feature efficiency measure for each
            feature.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           (Cited on page 7). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        """
        if ovo_comb is None or cls_inds is None or class_freqs is None:
            sub_dic = cls.precompute_complexity(y=y)
            ovo_comb = sub_dic["ovo_comb"]
            cls_inds = sub_dic["cls_inds"]
            class_freqs = sub_dic["class_freqs"]

        _ovo_comb = np.asarray(ovo_comb, dtype=int)
        _class_freqs = np.asarray(class_freqs, dtype=int)
        _cls_inds = np.asarray(cls_inds, dtype=bool)

        f4 = np.zeros(_ovo_comb.shape[0], dtype=float)

        for ind, (cls_id_1, cls_id_2) in enumerate(_ovo_comb):
            cls_subset_union = np.logical_or(
                _cls_inds[cls_id_1, :], _cls_inds[cls_id_2, :]
            )

            cls_1 = _cls_inds[cls_id_1, cls_subset_union]
            cls_2 = _cls_inds[cls_id_2, cls_subset_union]
            N_subset = N[cls_subset_union, :]

            # Search only on remaining features, without copying any data
            valid_attr_inds = np.arange(N_subset.shape[1])
            N_view = N_subset[:, valid_attr_inds]

            while N_view.size > 0:
                N_cls_1, N_cls_2 = N_view[cls_1, :], N_view[cls_2, :]

                # Note: 'feat_overlapped_region' is a boolean vector with
                # True values if the example is in the overlapping region
                (ind_less_overlap, _, feat_overlapped_region,) = cls._calc_overlap(
                    N=N_view,
                    minmax=cls._calc_minmax(N_cls_1, N_cls_2),
                    maxmin=cls._calc_maxmin(N_cls_1, N_cls_2),
                )

                # Boolean that if True, this example is in the overlapping
                # region
                overlapped_region = feat_overlapped_region[:, ind_less_overlap]

                # Removing the non-overlapping instances
                N_subset = N_subset[overlapped_region, :]
                cls_1 = cls_1[overlapped_region]
                cls_2 = cls_2[overlapped_region]

                # Removing the most efficient feature
                # Note: previous versions used to delete it directly from data
                # 'N_subset', but that procedure takes up much more memory
                # because each 'np.delete' operation creates a new dataset.
                valid_attr_inds = np.delete(valid_attr_inds, ind_less_overlap)
                N_view = N_subset[:, valid_attr_inds]

            subset_size = N_subset.shape[0]

            f4[ind] = subset_size / (_class_freqs[cls_id_1] + _class_freqs[cls_id_2])

        # The measure is computed in the literature using the mean. However, it
        # is formulated here as a meta-feature. Therefore, the post-processing
        # should be used to get the mean and other measures as well.
        return f4

    @classmethod
    def ft_l1(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        ovo_comb: t.Optional[np.ndarray] = None,
        cls_inds: t.Optional[np.ndarray] = None,
        class_freqs: t.Optional[np.ndarray] = None,
        svc_pipeline: t.Optional[sklearn.pipeline.Pipeline] = None,
        max_iter: t.Union[int, float] = 1e5,
        random_state: t.Optional[int] = None,
    ) -> np.ndarray:
        """Sum of error distance by linear programming.

        This measure assesses if the data are linearly separable by
        computing, for a dataset, the sum of the distances of incorrectly
        classified examples to a linear boundary used in their classification.
        If the value of L1 is zero, then theproblem is linearly separable and
        can be considered simpler than a problem for which a non-linear
        boundary is required.

        The average value of this measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        ovo_comb : :obj:`np.ndarray`, optional
            List of all class OVO combination, i.e., all combinations of
            distinct class indices by pairs ([(0, 1), (0, 2) ...].)

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows corresponds to each distinct class, and the columns
            corresponds to the instances.

        class_freqs : :obj:`np.ndarray`, optional
            The number of examples in each class. The indices corresponds to
            the classes.

        max_iter : float or int, optional
            Maximum number of iterations allowed for the support vector
            machine model convergence. This parameter can receive float
            numbers to be compatible with the Python scientific notation
            data type. Used only if ``svc_pipeline`` is None.

        svc_pipeline : :obj:`sklearn.pipeline.Pipeline`, optional
            Support Vector Classifier learning pipeline. Traditionally, the
            pipeline used is a data standardization (mean = 0 and variance = 1)
            before the learning model, which is a Support Vector Classifier
            (linear kernel.) However, any variation of this pipeline can also
            be used. Note that this metafeature is formulated using a linear
            classifier. If this argument is none, the described pipeline
            (standardization + SVC) is used by default.

        random_state : int, optional
            Random seed for dual coordinate descent while fitting the
            Support Vector Classifier model. Check `sklearn.svm.LinearSVC`
            documentation (`random_state` parameter) for more information.
            Used only if ``svc_pipeline`` is None.

        Returns
        -------
        :obj:`np.ndarray`
            Complement of the inverse of the sum of distances from a Support
            Vector Classifier (SVC) hyperplane of incorrectly classified
            instances.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           (Cited on page 9). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        """
        if ovo_comb is None or cls_inds is None or class_freqs is None:
            sub_dic = cls.precompute_complexity(y=y)
            ovo_comb = sub_dic["ovo_comb"]
            cls_inds = sub_dic["cls_inds"]
            class_freqs = sub_dic["class_freqs"]

        _ovo_comb = np.asarray(ovo_comb, dtype=int)
        _class_freqs = np.asarray(class_freqs, dtype=int)
        _cls_inds = np.asarray(cls_inds, dtype=bool)

        if svc_pipeline is None:
            sub_dic = cls.precompute_complexity_svm(
                max_iter=max_iter, y=y, random_state=random_state
            )

            svc_pipeline = sub_dic["svc_pipeline"]

        sum_err_dist = np.zeros(_ovo_comb.shape[0], dtype=float)

        for ind, (cls_1, cls_2) in enumerate(_ovo_comb):
            cls_union = np.logical_or(_cls_inds[cls_1, :], _cls_inds[cls_2, :])

            N_subset = N[cls_union, :]
            y_subset = _cls_inds[cls_1, cls_union]

            svc_pipeline.fit(N_subset, y_subset)
            y_pred = svc_pipeline.predict(N_subset)
            misclassified_insts = N_subset[y_pred != y_subset, :]

            if misclassified_insts.size:
                insts_dists = svc_pipeline.decision_function(misclassified_insts)

            else:
                insts_dists = np.array([0.0], dtype=float)

            sum_err_dist[ind] = np.linalg.norm(insts_dists, ord=1) / (
                _class_freqs[cls_1] + _class_freqs[cls_2]
            )

        l1 = 1.0 - 1.0 / (1.0 + sum_err_dist)
        # The measure is computed in the literature using the mean. However, it
        # is formulated here as a meta-feature. Therefore, the post-processing
        # should be used to get the mean and other measures as well.
        return l1

    @classmethod
    def ft_l2(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        ovo_comb: t.Optional[np.ndarray] = None,
        cls_inds: t.Optional[np.ndarray] = None,
        svc_pipeline: t.Optional[sklearn.pipeline.Pipeline] = None,
        max_iter: t.Union[int, float] = 1e5,
        random_state: t.Optional[int] = None,
    ) -> np.ndarray:
        """Compute the OVO subsets error rate of linear classifier.

        The linear model used is induced by the Support Vector
        Machine algorithm.

        The average value of this measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        ovo_comb : :obj:`np.ndarray`, optional
            List of all class OVO combination, i.e., all combinations of
            distinct class indices by pairs ([(0, 1), (0, 2) ...].)

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows corresponds to each distinct class, and the columns
            corresponds to the instances.

        svc_pipeline : :obj:`sklearn.pipeline.Pipeline`, optional
            Support Vector Classifier learning pipeline. Traditionally, the
            pipeline used is a data standardization (mean = 0 and variance = 1)
            before the learning model, which is a Support Vector Classifier
            (linear kernel.) However, any variation of this pipeline can also
            be used. Note that this metafeature is formulated using a linear
            classifier. If this argument is none, the described pipeline
            (standardization + SVC) is used by default.

        max_iter : float or int, optional
            Maximum number of iterations allowed for the support vector
            machine model convergence. This parameter can receive float
            numbers to be compatible with the Python scientific notation
            data type. Used only if ``svc_pipeline`` is None.

        random_state : int, optional
            Random seed for dual coordinate descent while fitting the
            Support Vector Classifier model. Check `sklearn.svm.LinearSVC`
            documentation (`random_state` parameter) for more information.
            Used only if ``svc_pipeline`` is None.

        Returns
        -------
        :obj:`np.ndarray`
            An array with the collective error rate of linear classifier
            measure for each OVO subset.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           (Cited on page 9). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        """
        if ovo_comb is None or cls_inds is None:
            sub_dic = cls.precompute_complexity(y=y)
            ovo_comb = sub_dic["ovo_comb"]
            cls_inds = sub_dic["cls_inds"]

        _ovo_comb = np.asarray(ovo_comb, dtype=int)
        _cls_inds = np.asarray(cls_inds, dtype=bool)

        if svc_pipeline is None:
            sub_dic = cls.precompute_complexity_svm(
                max_iter=max_iter, y=y, random_state=random_state
            )

            svc_pipeline = sub_dic["svc_pipeline"]

        l2 = np.zeros(_ovo_comb.shape[0], dtype=float)

        for ind, (cls_1, cls_2) in enumerate(_ovo_comb):
            cls_union = np.logical_or(_cls_inds[cls_1, :], _cls_inds[cls_2, :])

            N_subset = N[cls_union, :]
            y_subset = _cls_inds[cls_1, cls_union]

            svc_pipeline.fit(N_subset, y_subset)
            y_pred = svc_pipeline.predict(N_subset)

            error = sklearn.metrics.zero_one_loss(
                y_true=y_subset, y_pred=y_pred, normalize=True
            )

            l2[ind] = error

        # The measure is computed in the literature using the mean. However, it
        # is formulated here as a meta-feature. Therefore, the post-processing
        # should be used to get the mean and other measures as well.
        return l2

    @classmethod
    def ft_l3(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        ovo_comb: t.Optional[np.ndarray] = None,
        cls_inds: t.Optional[np.ndarray] = None,
        svc_pipeline: t.Optional[sklearn.pipeline.Pipeline] = None,
        max_iter: t.Union[int, float] = 1e5,
        random_state: t.Optional[int] = None,
    ) -> np.ndarray:
        """Non-Linearity of a linear classifier.

        This index is sensitive to how the data from a class are
        distributed inthe border regions and also on how much the convex
        hulls which delimit the classes overlap. In particular, it detects
        the presence of concavities in the class boundaries. Higher values
        indicate a greater complexity.

        The average value of this measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        ovo_comb : :obj:`np.ndarray`, optional
            List of all class OVO combination, i.e., all combinations of
            distinct class indices by pairs ([(0, 1), (0, 2) ...].)

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows corresponds to each distinct class, and the columns
            corresponds to the instances.

        svc_pipeline : :obj:`sklearn.pipeline.Pipeline`, optional
            Support Vector Classifier learning pipeline. Traditionally, the
            pipeline used is a data standardization (mean = 0 and variance = 1)
            before the learning model, which is a Support Vector Classifier
            (linear kernel.) However, any variation of this pipeline can also
            be used. Note that this metafeature is formulated using a linear
            classifier. If this argument is none, the described pipeline
            (standardization + SVC) is used by default.

        max_iter : float or int, optional
            Maximum number of iterations allowed for the support vector
            machine model convergence. This parameter can receive float
            numbers to be compatible with the Python scientific notation
            data type. Used only if ``svc_pipeline`` is None.

        random_state : int, optional
            Random seed for dual coordinate descent while fitting the
            Support Vector Classifier model. Check `sklearn.svm.LinearSVC`
            documentation (`random_state` parameter) for more information.
            Used only if ``svc_pipeline`` is None.

        Returns
        -------
        :obj:`np.ndarray`
            Zero-one losses of a Support Vector Classifier for a randomly
            interpolated dataset using the original instances. The classes
            are separated in a OVO (One-Versus-One) fashion.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           (Cited on page 9). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        """
        if ovo_comb is None or cls_inds is None:
            sub_dic = cls.precompute_complexity(y=y)
            ovo_comb = sub_dic["ovo_comb"]
            cls_inds = sub_dic["cls_inds"]

        _ovo_comb = np.asarray(ovo_comb, dtype=int)
        _cls_inds = np.asarray(cls_inds, dtype=bool)

        if svc_pipeline is None:
            sub_dic = cls.precompute_complexity_svm(
                max_iter=max_iter, y=y, random_state=random_state
            )

            svc_pipeline = sub_dic["svc_pipeline"]

        l3 = np.zeros(_ovo_comb.shape[0], dtype=float)

        for ind, (cls_1, cls_2) in enumerate(_ovo_comb):
            cls_union = np.logical_or(_cls_inds[cls_1, :], _cls_inds[cls_2, :])

            N_subset = N[cls_union, :]
            y_subset = _cls_inds[cls_1, cls_union]
            _cls_inds_subset = _cls_inds[:, cls_union]

            svc_pipeline.fit(N_subset, y_subset)

            # Note: changing 'random_state' every iteration to prevent
            # the same subset of instances to be selected for the same
            # class over and over again. Also note that this does not
            # breaks the deterministic result of the calculations.
            if random_state is not None:
                random_state += 1

            N_interpol, y_interpol = cls._interpolate(
                N=N_subset,
                y=y_subset,
                cls_inds=_cls_inds_subset,
                random_state=random_state,
            )

            y_pred = svc_pipeline.predict(N_interpol)

            error = sklearn.metrics.zero_one_loss(
                y_true=y_interpol, y_pred=y_pred, normalize=True
            )

            l3[ind] = error

        # The measure is computed in the literature using the mean. However, it
        # is formulated here as a meta-feature. Therefore, the post-processing
        # should be used to get the mean and other measures as well.
        return l3

    @classmethod
    def ft_n1(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        metric: str = "gower",
        p: t.Union[int, float] = 2,
        N_scaled: t.Optional[np.ndarray] = None,
        norm_dist_mat: t.Optional[np.ndarray] = None,
    ) -> float:
        """Compute the fraction of borderline points.

        This measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        metric : str, optional
            Metric used to calculate the distances between the instances.
            Check the ``scipy.spatial.distance.cdist`` documentation to
            get a list of all available metrics. This argument is used
            only if ``norm_dist_mat`` is None.

        p : int, optional
            Power parameter for the Minkowski metric. When p = 1, this is
            equivalent to using Manhattan distance (l1), and Euclidean
            distance (l2) for p = 2. For arbitrary p, Minkowski distance
            (l_p) is used. Used only if ``norm_dist_mat`` is None.

        N_scaled : :obj:`np.ndarray`, optional
            Numerical data ``N`` with each feature normalized  in [0, 1]
            range. Used only if ``norm_dist_mat`` is None. Used to take
            advantage of precomputations.

        norm_dist_mat : :obj:`np.ndarray`, optional
            Square matrix with the pairwise distances between each
            instance in ``N_scaled``, i.e., between the normalized
            instances. Used to take advantage of precomputations.

        Returns
        -------
        float
            Fraction of borderline points.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           (Cited on page 9-10). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        """
        if norm_dist_mat is None:
            norm_dist_mat, _, _ = cls._calc_norm_dist_mat(
                N=N, metric=metric, p=p, N_scaled=N_scaled
            )

        _norm_dist_mat = np.asfarray(norm_dist_mat)

        # Compute the minimum spanning tree using Kruskal's Minimum
        # Spanning Tree algorithm.
        # Note: in the paper, the authors used Prim's algorithm.
        # Our implementation may change it in a future version due to
        # time complexity advantages of Prim's algorithm in this context.
        mst = scipy.sparse.csgraph.minimum_spanning_tree(
            csgraph=np.triu(_norm_dist_mat, k=1), overwrite=True
        )

        node_id_i, node_id_j = np.nonzero(mst)

        # Which edges have nodes with different class
        which_have_diff_cls = y[node_id_i] != y[node_id_j]

        # Number of vertices connected with different classes
        borderline_inst_num = np.unique(
            np.concatenate(
                [
                    node_id_i[which_have_diff_cls],
                    node_id_j[which_have_diff_cls],
                ]
            )
        ).size

        n1 = borderline_inst_num / y.size

        return n1

    @classmethod
    def ft_n2(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        metric: str = "gower",
        p: t.Union[int, float] = 2,
        class_freqs: t.Optional[np.ndarray] = None,
        cls_inds: t.Optional[np.ndarray] = None,
        N_scaled: t.Optional[np.ndarray] = None,
        norm_dist_mat: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Ratio of intra and extra class nearest neighbor distance.

        This measure computes the ratio of two sums:
            - The sum of the distances between each example and its closest
              neighborfrom the same class (intra-class); and
            - The sum of the distances between each example and its closest
              neighbor fromanother class (extra-class)

        The average value of this measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        metric : str, optional
            Metric used to calculate the distances between the instances.
            Check the ``scipy.spatial.distance.cdist`` documentation to
            get a list of all available metrics. This argument is used
            only if ``norm_dist_mat`` is None.

        p : int, optional
            Power parameter for the Minkowski metric. When p = 1, this is
            equivalent to using Manhattan distance (l1), and Euclidean
            distance (l2) for p = 2. For arbitrary p, Minkowski distance
            (l_p) is used. Used only if ``norm_dist_mat`` is None.

        class_freqs : :obj:`np.ndarray`, optional
            The number of examples in each class. The indices corresponds to
            the classes.

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows corresponds to each distinct class, and the columns
            corresponds to the instances.

        N_scaled : :obj:`np.ndarray`, optional
            Numerical data ``N`` with each feature normalized  in [0, 1]
            range. Used only if ``norm_dist_mat`` is None. Used to take
            advantage of precomputations.

        norm_dist_mat : :obj:`np.ndarray`, optional
            Square matrix with the pairwise distances between each
            instance in ``N_scaled``, i.e., between the normalized
            instances. Used to take advantage of precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Complement of the inverse of the intra and extra class variance.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           (Cited on page 9). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        """
        classes = None

        if class_freqs is None:
            classes, class_freqs = np.unique(y, return_counts=True)

        if cls_inds is None:
            cls_inds = _utils.calc_cls_inds(y, classes)

        if norm_dist_mat is None:
            norm_dist_mat, _, _ = cls._calc_norm_dist_mat(
                N=N, metric=metric, p=p, N_scaled=N_scaled
            )

        intra_extra = np.zeros(y.size, dtype=float)

        cur_ind = 0

        for cls_ind, inds_cur_cls in enumerate(cls_inds):
            norm_dist_mat_intracls = norm_dist_mat[inds_cur_cls, :][:, inds_cur_cls]
            norm_dist_mat_intercls = norm_dist_mat[~inds_cur_cls, :][:, inds_cur_cls]

            norm_dist_mat_intracls[
                np.diag_indices_from(norm_dist_mat_intracls)
            ] = np.inf

            _aux = np.arange(class_freqs[cls_ind])

            intra = norm_dist_mat_intracls[
                np.argmin(norm_dist_mat_intracls, axis=0), _aux
            ]
            extra = norm_dist_mat_intercls[
                np.argmin(norm_dist_mat_intercls, axis=0), _aux
            ]

            next_ind = cur_ind + class_freqs[cls_ind]
            intra_extra[cur_ind:next_ind] = intra / extra
            cur_ind = next_ind

        # Note: in the original paper, 'intra_extra' is the ratio of two
        # sums. However, to enable summarization, the sums are omitted.
        n2 = 1.0 - 1.0 / (1.0 + intra_extra)

        return n2

    @classmethod
    def ft_n3(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        metric: str = "gower",
        p: t.Union[int, float] = 2,
        N_scaled: t.Optional[np.ndarray] = None,
        norm_dist_mat: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Error rate of the nearest neighbor classifier.

        The N3 measure refers to the error rate of a 1-NN classifier
        that is estimated using a leave-one-out cross-validation procedure.

        The average value of this measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        metric : str, optional
            Metric used to calculate the distances between the instances.
            Check the ``scipy.spatial.distance.cdist`` documentation to
            get a list of all available metrics. This argument is used
            only if ``norm_dist_mat`` is None.

        p : int, optional
            Power parameter for the Minkowski metric. When p = 1, this is
            equivalent to using Manhattan distance (l1), and Euclidean
            distance (l2) for p = 2. For arbitrary p, Minkowski distance
            (l_p) is used.

        N_scaled : :obj:`np.ndarray`, optional
            Numerical data ``N`` with each feature normalized  in [0, 1]
            range. Used only if ``norm_dist_mat`` is None. Used to take
            advantage of precomputations.

        norm_dist_mat : :obj:`np.ndarray`, optional
            Square matrix with the pairwise distances between each
            instance in ``N_scaled``, i.e., between the normalized
            instances. Used to take advantage of precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Binary array of misclassification of a 1-NN classifier.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           (Cited on page 9). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        """
        if norm_dist_mat is None:
            norm_dist_mat, _, _ = cls._calc_norm_dist_mat(
                N=N, metric=metric, p=p, N_scaled=N_scaled
            )

        model = sklearn.neighbors.KNeighborsClassifier(
            n_neighbors=1, metric="precomputed"
        ).fit(norm_dist_mat, y)

        neighbor_inds = model.kneighbors(return_distance=False).ravel()

        misclassifications = np.not_equal(y[neighbor_inds], y).astype(int)

        # The measure is computed in the literature using the mean. However, it
        # is formulated here as a meta-feature. Therefore, the post-processing
        # should be used to get the mean and other measures as well.
        return misclassifications

    @classmethod
    def ft_n4(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        metric: str = "gower",
        p: t.Union[int, float] = 2,
        n_neighbors: int = 1,
        random_state: t.Optional[int] = None,
        cls_inds: t.Optional[np.ndarray] = None,
        N_scaled: t.Optional[np.ndarray] = None,
        norm_dist_mat: t.Optional[np.ndarray] = None,
        orig_dist_mat_min: t.Optional[float] = None,
        orig_dist_mat_ptp: t.Optional[float] = None,
    ) -> np.ndarray:
        """Compute the non-linearity of the k-NN Classifier.

        The average value of this measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        metric : str, optional
            The distance metric used in the internal kNN classifier. See the
            documentation of the ``scipy.spatial.distance.cdist`` class
            for a list of available metrics. Used only if ``norm_dist_mat``
            is None.

        p : int, optional
            Power parameter for the Minkowski metric. When p = 1, this is
            equivalent to using Manhattan distance (l1), and Euclidean
            distance (l2) for p = 2. For arbitrary p, Minkowski distance
            (l_p) is used. Used only if ``norm_dist_mat`` is None.

        n_neighbors : int, optional
            Number of neighbors used for the Nearest Neighbors classifier.

        random_state : int, optional
            If given, set the random seed before computing the randomized
            data interpolation.

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows corresponds to each distinct class, and the columns
            corresponds to the instances. Used to take advantages of
            precomputations.

        N_scaled : :obj:`np.ndarray`, optional
            Numerical data ``N`` with each feature normalized  in [0, 1]
            range. Used only if ``norm_dist_mat`` is None. Used to take
            advantage of precomputations.

        norm_dist_mat : :obj:`np.ndarray`, optional
            Square matrix with the pairwise distances between each
            instance in ``N_scaled``, i.e., between the normalized
            instances. Used to take advantage of precomputations.
            Used if and only if ``orig_dist_mat_min`` AND
            ``orig_dist_mat_ptp`` are also given (non None).

        orig_dist_mat_min : :obj:`float`, optional
            Minimal distance between the original instances in ``N``.

        orig_dist_mat_ptp : :obj:`float`, optional
            Range (max - min) of distances between the original instances
            in ``N``.

        Returns
        -------
        :obj:`np.ndarray`
            Misclassifications of the k-NN classifier in the interpolated
            dataset.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           (Cited on page 9-11). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        """
        if cls_inds is None:
            cls_inds = _utils.calc_cls_inds(y)

        if N_scaled is None:
            N_scaled = cls._scale_N(N=N)

        if (
            norm_dist_mat is None
            or orig_dist_mat_min is None
            or orig_dist_mat_ptp is None
        ):
            (
                norm_dist_mat,
                orig_dist_mat_min,
                orig_dist_mat_ptp,
            ) = cls._calc_norm_dist_mat(N=N, metric=metric, p=p, N_scaled=N_scaled)

        N_interpol, y_interpol = cls._interpolate(
            N=N_scaled, y=y, cls_inds=cls_inds, random_state=random_state
        )

        knn = sklearn.neighbors.KNeighborsClassifier(
            n_neighbors=n_neighbors, metric="precomputed"
        ).fit(norm_dist_mat, y)

        if metric == "gower":
            test_dist = gower.gower_matrix(N_interpol, N_scaled)

        else:
            test_dist = scipy.spatial.distance.cdist(
                N_interpol,
                N_scaled,
                metric=metric,
                p=p,
            )

        # Note: normalizing test data distances with original data
        # information in order to provide unbiased predictions (i.e.
        # avoid data leakage.)
        if np.not_equal(0.0, orig_dist_mat_ptp):
            test_dist = (test_dist - orig_dist_mat_min) / orig_dist_mat_ptp

        y_pred = knn.predict(test_dist)

        misclassifications = np.not_equal(y_interpol, y_pred).astype(int)

        # The measure is computed in the literature using the mean. However, it
        # is formulated here as a meta-feature. Therefore, the post-processing
        # should be used to get the mean and other measures as well.
        return misclassifications

    @classmethod
    def ft_c1(cls, y: np.ndarray, class_freqs: t.Optional[np.ndarray] = None) -> float:
        """Compute the entropy of class proportions.

        This measure is in [0, 1] range.

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Target attribute.

        class_freqs : :obj:`np.ndarray`, optional
            The number of examples in each class. The indices corresponds to
            the classes.

        Returns
        -------
        float
            Entropy of class proportions.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           (Cited on page 15). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        """
        if class_freqs is None:
            _, class_freqs = np.unique(y, return_counts=True)

        num_class = class_freqs.size

        # Note: calling 'ft_nre' just to make explicity the link
        # between this metafeature and 'C1'.
        c1 = MFEClustering.ft_nre(y=y, class_freqs=class_freqs) / np.log(num_class)

        return c1

    @classmethod
    def ft_c2(cls, y: np.ndarray, class_freqs: t.Optional[np.ndarray] = None) -> float:
        """Compute the imbalance ratio.

        This measure is in [0, 1] range.

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Target attribute.

        class_freqs : :obj:`np.ndarray`, optional
            The number of examples in each class. The indices corresponds to
            the classes.

        Returns
        -------
        float
            The imbalance ratio.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           (Cited on page 16). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        """
        if class_freqs is None:
            _, class_freqs = np.unique(y, return_counts=True)

        num_inst = y.size
        num_class = class_freqs.size

        aux = (num_class - 1) / num_class
        imbalance_ratio = aux * np.sum(class_freqs / (num_inst - class_freqs))

        c2 = 1 - 1 / imbalance_ratio

        return c2

    @classmethod
    def ft_t1(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        metric: str = "gower",
        p: t.Union[int, float] = 2,
        cls_inds: t.Optional[np.ndarray] = None,
        N_scaled: t.Optional[np.ndarray] = None,
        norm_dist_mat: t.Optional[np.ndarray] = None,
        orig_dist_mat_min: t.Optional[float] = None,
        orig_dist_mat_ptp: t.Optional[float] = None,
    ) -> np.ndarray:
        """Fraction of hyperspheres covering data.

        This measure uses a process that builds hyperspheres centered
        at each one of the examples. In this implementation, we stop the
        growth of the hypersphere when the hyperspheres centered at two
        points of opposite classes just start to touch.

        Once the radiuses of all hyperspheres are found, a post-processing
        step can be applied to verify which hyperspheres must be absorbed
        (all hyperspheres completely within larger hyperspheres.)

        This measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        metric : str, optional
            Metric used to calculate the distances between the instances.
            Check the ``scipy.spatial.distance.cdist`` documentation to
            get a list of all available metrics. This argument is used
            only if ``norm_dist_mat`` is None.

        p : int, optional
            Power parameter for the Minkowski metric. When p = 1, this is
            equivalent to using Manhattan distance (l1), and Euclidean
            distance (l2) for p = 2. For arbitrary p, Minkowski distance
            (l_p) is used. Used only if ``norm_dist_mat`` is None.

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows corresponds to each distinct class, and the columns
            corresponds to the instances. Used only if the arguments
            ``nearest_enemy_dist`` or ``nearest_enemy_ind`` are None.

        N_scaled : :obj:`np.ndarray`, optional
            Numerical data ``N`` with each feature normalized  in [0, 1]
            range. Used only if ``norm_dist_mat`` is None. Used to take
            advantage of precomputations.

        norm_dist_mat : :obj:`np.ndarray`, optional
            Square matrix with the pairwise distances between each
            instance in ``N_scaled``, i.e., between the normalized
            instances. Used to take advantage of precomputations.
            Used if and only if ``orig_dist_mat_min`` and
            ``orig_dist_mat_ptp`` are also given (non None).

        orig_dist_mat_min : :obj:`float`, optional
            Minimal distance between the original instances in ``N``.

        orig_dist_mat_ptp : :obj:`float`, optional
            Range (max - min) of distances between the original instances
            in ``N``.

        Returns
        -------
        :obj:`np.ndarray`
            Array with the fraction of instances inside each remaining
            hypersphere.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           (Cited on page 9). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        .. [2] Tin K Ho and Mitra Basu. Complexity measures of supervised
           classification problems. IEEE Transactions on Pattern Analysis and
           Machine Intelligence, 24(3):289–300, 2002.
        """

        def _calc_hyperspheres_radius(
            nearest_enemy_ind: np.ndarray, nearest_enemy_dist: np.ndarray
        ) -> np.ndarray:
            """Get the radius of hyperspheres which cover the given dataset."""

            def _recurse_radius_calc(ind_inst: int) -> float:
                """Recursively calculate hyperspheres to cover dataset."""
                if radius[ind_inst] >= 0.0:
                    return radius[ind_inst]

                ind_enemy = nearest_enemy_ind[ind_inst]

                if ind_inst == nearest_enemy_ind[ind_enemy]:
                    # In this case, both instances are the nearest enemy of
                    # each other, thus the hypersphere radius of both is
                    # half the distance between the instances.
                    radius[ind_enemy] = radius[ind_inst] = (
                        0.5 * nearest_enemy_dist[ind_inst]
                    )
                    return radius[ind_inst]

                # Note: set the current instance radius to '0' before
                # recursive call to prevent (uncommon cases of) infinite
                # recursion
                radius[ind_inst] = 0.0

                radius_enemy = _recurse_radius_calc(ind_inst=ind_enemy)

                radius[ind_inst] = abs(nearest_enemy_dist[ind_inst] - radius_enemy)

                return radius[ind_inst]

            radius = np.full(nearest_enemy_ind.size, fill_value=-1.0, dtype=float)

            for ind in np.arange(radius.size):
                if radius[ind] < 0.0:
                    _recurse_radius_calc(ind_inst=ind)

            return radius

        def _is_hypersphere_in(
            center_a: np.ndarray,
            center_b: np.ndarray,
            radius_a: float,
            radius_b: float,
        ) -> bool:
            """Checks if a hypersphere `a` is in a hypersphere `b`."""
            upper_a, lower_a = center_a + radius_a, center_a - radius_a
            upper_b, lower_b = center_b + radius_b, center_b - radius_b
            for ind in np.arange(center_a.size):
                if (upper_a[ind] > upper_b[ind]) or (lower_a[ind] < lower_b[ind]):
                    return False

            return True

        def _agglomerate_hyperspheres(
            centers: np.ndarray, radius: np.ndarray
        ) -> np.ndarray:
            """Agglomerate internal hyperspheres into outer hyperspheres.

            Returns the number of training instances within each
            remaining hypersphere. Interal hyperspheres will have
            zero instances within.
            """
            sorted_sphere_inds = np.argsort(radius)
            sphere_inst_num = np.ones(radius.size, dtype=int)

            for ind_a, ind_sphere_a in enumerate(sorted_sphere_inds[:-1]):
                for ind_sphere_b in sorted_sphere_inds[:ind_a:-1]:
                    if _is_hypersphere_in(
                        center_a=centers[ind_sphere_a, :],
                        center_b=centers[ind_sphere_b, :],
                        radius_a=radius[ind_sphere_a],
                        radius_b=radius[ind_sphere_b],
                    ):

                        sphere_inst_num[ind_sphere_b] += sphere_inst_num[ind_sphere_a]
                        sphere_inst_num[ind_sphere_a] = 0
                        break

            return sphere_inst_num

        if N_scaled is None:
            N_scaled = cls._scale_N(N=N)

        if cls_inds is None:
            cls_inds = _utils.calc_cls_inds(y)

        if (
            norm_dist_mat is None
            or orig_dist_mat_min is None
            or orig_dist_mat_ptp is None
        ):
            orig_dist_mat, _, _ = cls._calc_norm_dist_mat(
                N=N, metric=metric, p=p, N_scaled=N_scaled, normalize=False
            )

        else:
            orig_dist_mat = norm_dist_mat * orig_dist_mat_ptp + orig_dist_mat_min

        # Note: using the original pairwise distances between instances,
        # instead of the normalized ones, to preserve geometrical/spatial
        # coherence between the sphere centers, radius, and placements.
        # That is why we are not using neither the precomputed
        # 'nearest_enemy_dist' nor 'nearest_enemy_ind' values here.
        nearest_enemy_dist, nearest_enemy_ind = cls._calc_nearest_enemies(
            norm_dist_mat=orig_dist_mat, cls_inds=cls_inds
        )

        radius = _calc_hyperspheres_radius(
            nearest_enemy_ind=nearest_enemy_ind,
            nearest_enemy_dist=nearest_enemy_dist,
        )

        sphere_inst_count = _agglomerate_hyperspheres(centers=N_scaled, radius=radius)

        # Note: in the reference paper, just the fraction of
        # remaining hyperspheres to the size of the dataset is
        # calculated. However, just like the R ECoL package,
        # we return the fraction of the number of instances in
        # each remaining hypersphere to provide more informative
        # summarization values.
        t1 = sphere_inst_count[sphere_inst_count > 0] / y.size

        return t1

    @classmethod
    def ft_t2(cls, N: np.ndarray) -> float:
        """Compute the average number of features per dimension.

        This measure is in (0, m] range, where `m` is the number of
        features in ``N``.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numeric attributes from fitted data.

        Returns
        -------
        float
            Average number of features per dimension.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           (Cited on page 15). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        """
        # Note: This metafeature has a link with the 'ft_attr_to_inst',
        # but considers the trandormed data in N instead of the original
        # attributes 'X'. Maybe this is a aspect that may change in the
        # future.
        # return MFEGeneral.ft_attr_to_inst(X=X)

        return N.shape[1] / N.shape[0]

    @classmethod
    def ft_t3(
        cls,
        N: np.ndarray,
        num_attr_pca: t.Optional[int] = None,
        random_state: t.Optional[int] = None,
    ) -> float:
        """Compute the average number of PCA dimensions per points.

        This measure is in (0, m] range, where `m` is the number of
        features in ``N``.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        num_attr_pca : int, optional
            Number of features after PCA where a fraction of at least 0.95
            of the data variance is explained by the selected components.

        random_state : int, optional
            If the fitted data is huge and the number of principal components
            to be kept is low, then the PCA analysis is done using a randomized
            strategy for efficiency. This random seed keeps the results
            replicable. Check ``sklearn.decomposition.PCA`` documentation for
            more information.

        Returns
        -------
        float
            Average number of PCA dimensions (explaining at least 95% of the
            data variance) per points.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           (Cited on page 15). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        """
        if num_attr_pca is None:
            sub_dic = cls.precompute_pca_tx(N=N, random_state=random_state)
            num_attr_pca = sub_dic["num_attr_pca"]

        num_inst = N.shape[0]

        return num_attr_pca / num_inst

    @classmethod
    def ft_t4(
        cls,
        N: np.ndarray,
        num_attr_pca: t.Optional[int] = None,
        random_state: t.Optional[int] = None,
    ) -> float:
        """Compute the ratio of the PCA dimension to the original dimension.

        The components kept in the PCA dimension explains at least 95% of
        the data variance.

        This measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        num_attr_pca : int, optional
            Number of features after PCA where a fraction of at least 0.95
            of the data variance is explained by the selected components.

        random_state : int, optional
            If the fitted data is huge and the number of principal components
            to be kept is low, then the PCA analysis is done using a randomized
            strategy for efficiency. This random seed keeps the results
            replicable. Check ``sklearn.decomposition.PCA`` documentation for
            more information.

        Returns
        -------
        float
            Ratio of the PCA dimension (explaining at least 95% of the data
            variance) to the original dimension.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           (Cited on page 15). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        """
        if num_attr_pca is None:
            sub_dic = cls.precompute_pca_tx(N=N, random_state=random_state)
            num_attr_pca = sub_dic["num_attr_pca"]

        num_attr = N.shape[1]

        return num_attr_pca / num_attr

    @classmethod
    def ft_lsc(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        metric: str = "gower",
        p: t.Union[int, float] = 2,
        cls_inds: t.Optional[np.ndarray] = None,
        N_scaled: t.Optional[np.ndarray] = None,
        norm_dist_mat: t.Optional[np.ndarray] = None,
        nearest_enemy_dist: t.Optional[np.ndarray] = None,
    ) -> float:
        """Local set average cardinality.

        The Local-Set (LS) of an example `x_i` in a dataset ``N`` is
        defined as the set of points from ``N`` whose distance to `x_i`
        is smaller than the distance from `x_i` and its nearest enemy
        (the nearest instance from a distinct class of `x_i`.)

        The cardinality of the LS of an example indicates its proximity
        to the decision boundary and also the narrowness of the gap
        between the classes.

        This measure is in [0, 1 - 1/n] range, where `n` is the number
        of instances in ``N``.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        metric : str, optional
            Metric used to calculate the distances between the instances.
            Check the ``scipy.spatial.distance.cdist`` documentation to
            get a list of all available metrics. This argument is used
            only if ``norm_dist_mat`` is None.

        p : int, optional
            Power parameter for the Minkowski metric. When p = 1, this is
            equivalent to using Manhattan distance (l1), and Euclidean
            distance (l2) for p = 2. For arbitrary p, Minkowski distance
            (l_p) is used. Used only if ``norm_dist_mat`` is None.

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows corresponds to each distinct class, and the columns
            corresponds to the instances. Used only if the argument
            ``nearest_enemy_dist`` is None.

        N_scaled : :obj:`np.ndarray`, optional
            Numerical data ``N`` with each feature normalized  in [0, 1]
            range. Used only if ``norm_dist_mat`` is None. Used to take
            advantage of precomputations.

        norm_dist_mat : :obj:`np.ndarray`, optional
            Square matrix with the pairwise distances between each
            instance in ``N_scaled``, i.e., between the normalized
            instances. Used to take advantage of precomputations.

        nearest_enemy_dist : :obj:`np.ndarray`, optional
            Distance of each instance to its nearest enemy (instances
            of a distinct class.)

        Returns
        -------
        float
            Local set average cardinality.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           (Cited on page 15). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        .. [2] Enrique Leyva, Antonio González, and Raúl Pérez. A set of
           complexity measures designed for applying meta-learning to instance
           selection. IEEE Transactions on Knowledge and Data Engineering,
           27(2):354–367, 2014.
        """
        if norm_dist_mat is None:
            norm_dist_mat, _, _ = cls._calc_norm_dist_mat(
                N=N, metric=metric, p=p, N_scaled=N_scaled
            )

        _norm_dist_mat = np.asfarray(norm_dist_mat)

        if nearest_enemy_dist is None:
            if cls_inds is None:
                cls_inds = _utils.calc_cls_inds(y)

            nearest_enemy_dist, _ = cls._calc_nearest_enemies(
                norm_dist_mat=_norm_dist_mat,
                cls_inds=cls_inds,
            )

        lsc = 1.0 - np.sum(_norm_dist_mat < nearest_enemy_dist) / (y.size**2)

        return float(lsc)

    @classmethod
    def ft_density(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        metric: str = "gower",
        p: float = 2.0,
        radius_frac: t.Union[int, float] = 0.15,
        n_jobs: t.Optional[int] = None,
        cls_inds: t.Optional[np.ndarray] = None,
        norm_dist_mat: t.Optional[np.ndarray] = None,
        adj_graph: t.Optional[igraph.Graph.Weighted_Adjacency] = None,
    ) -> float:
        """Average density of the network.

        This measure considers the number of edges that are retained in the
        graph (Same-class Radius Nearest Neighbors) built from the dataset
        normalized by the maximum number of edges between `y.size` instances.

        This measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        radius : float or int, optional
            Maximum distance between each pair of instances of the same
            class to both be considered neighbors of each other. Note that
            each feature of ``N`` is first normalized into the [0, 1]
            range before the neighbor calculations.

        metric : str, optional
            Metric used to calculate the distances between the instances.
            Check the ``scipy.spatial.distance.cdist`` documentation to
            get a list of all available metrics. This argument is used
            only if ``norm_dist_mat`` is None.

        p : int, optional
            Power parameter for the Minkowski metric. When p = 1, this is
            equivalent to using Manhattan distance (l1), and Euclidean
            distance (l2) for p = 2. For arbitrary p, Minkowski distance
            (l_p) is used. Used only if ``norm_dist_mat`` is None.

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows corresponds to each distinct class, and the columns
            corresponds to the instances.

        N_scaled : :obj:`np.ndarray`, optional
            Numerical data ``N`` with each feature normalized  in [0, 1]
            range. Used only if ``norm_dist_mat`` is None. Used to take
            advantage of precomputations.

        norm_dist_mat : :obj:`np.ndarray`, optional
            Square matrix with the pairwise distances between each
            instance in ``N_scaled``, i.e., between the normalized
            instances. Used to take advantage of precomputations.

        Returns
        -------
        float
            Complement of the ratio of total edges in the Radius Nearest
            Neighbors graph and the total number of edges that could
            possibly exists in a graph with the given number of instances.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           (Cited on page 9). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        """
        if adj_graph is None:
            adj_graph = cls._build_adjacency_graph(
                N=N,
                y=y,
                n_jobs=n_jobs,
                cls_inds=cls_inds,
                metric=metric,
                p=p,
                radius_frac=radius_frac,
                norm_dist_mat=norm_dist_mat,
            )

        density = 1.0 - adj_graph.density()

        return float(density)

    @classmethod
    def ft_cls_coef(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        metric: str = "gower",
        p: float = 2.0,
        radius_frac: t.Union[int, float] = 0.15,
        n_jobs: t.Optional[int] = None,
        cls_inds: t.Optional[np.ndarray] = None,
        norm_dist_mat: t.Optional[np.ndarray] = None,
        adj_graph: t.Optional[igraph.Graph.Weighted_Adjacency] = None,
    ) -> float:
        """Clustering coefficient.

        The clustering coefficient of a vertex `v_i` is given by the
        ratio of the number of edges between its neighbors (in a Same-class
        Radius Neighbor Graph) and the maximum number of edges that could
        possibly exist between them.

        This measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        radius : float or int, optional
            Maximum distance between each pair of instances of the same
            class to both be considered neighbors of each other. Note that
            each feature of ``N`` is first normalized into the [0, 1]
            range before the neighbor calculations.

        metric : str, optional
            Metric used to calculate the distances between the instances.
            Check the ``scipy.spatial.distance.cdist`` documentation to
            get a list of all available metrics. This argument is used
            only if ``norm_dist_mat`` is None.

        p : int, optional
            Power parameter for the Minkowski metric. When p = 1, this is
            equivalent to using Manhattan distance (l1), and Euclidean
            distance (l2) for p = 2. For arbitrary p, Minkowski distance
            (l_p) is used. Used only if ``norm_dist_mat`` is None.

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows corresponds to each distinct class, and the columns
            corresponds to the instances.

        N_scaled : :obj:`np.ndarray`, optional
            Numerical data ``N`` with each feature normalized  in [0, 1]
            range. Used only if ``norm_dist_mat`` is None. Used to take
            advantage of precomputations.

        norm_dist_mat : :obj:`np.ndarray`, optional
            Square matrix with the pairwise distances between each
            instance in ``N_scaled``, i.e., between the normalized
            instances. Used to take advantage of precomputations.

        Returns
        -------
        float
            Clustering coefficient of given data.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           (Cited on page 9). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        """
        if adj_graph is None:
            adj_graph = cls._build_adjacency_graph(
                N=N,
                y=y,
                n_jobs=n_jobs,
                cls_inds=cls_inds,
                metric=metric,
                p=p,
                radius_frac=radius_frac,
                norm_dist_mat=norm_dist_mat,
            )

        cls_coef = 1.0 - adj_graph.transitivity_undirected(mode="zero")

        return float(cls_coef)

    @classmethod
    def ft_hubs(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        metric: str = "gower",
        p: float = 2.0,
        radius_frac: t.Union[int, float] = 0.15,
        n_jobs: t.Optional[int] = None,
        cls_inds: t.Optional[np.ndarray] = None,
        norm_dist_mat: t.Optional[np.ndarray] = None,
        adj_graph: t.Optional[igraph.Graph.Weighted_Adjacency] = None,
    ) -> np.ndarray:
        """Hub score.

        The hub score scores each node by the number of connections it
        has to other nodes, weighted by the number of connections these
        neighbors have.

        The values of node hub score are given by the principal eigenvector
        of (A.t * A), where A is the adjacency matrix of the graph.

        The average value of this measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        metric : str, optional
            Metric used to calculate the distances between the instances.
            Check the ``scipy.spatial.distance.cdist`` documentation to
            get a list of all available metrics. Used only if `adj_graph` is None.

        p : int, optional
            Power parameter for the Minkowski metric. When p = 1, this is
            equivalent to using Manhattan distance (l1), and Euclidean
            distance (l2) for p = 2. For arbitrary p, Minkowski distance
            (l_p) is used. Used only if `adj_graph` is None.

        radius_frac : float or int, optional
            If `int`, maximum number of neighbors of the same class for each instance.
            If `float`, the maximum number of neighbors is computed as `radius_frac * len(N)`.
            Used only if `adj_graph` is None.

        n_jobs : int or None, optional
            Number of parallel processes to compute nearest neighbors.
            Used only if `adj_graph` is None.

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows corresponds to each distinct class, and the columns
            corresponds to the instances. Used only if `adj_graph` is None.

        norm_dist_mat: :obj:`np.ndarray`, optional
            Normalized distance matrix

        adj_graph : :obj:`igraph.Graph.Weighted_Adjacency`, optional
            Undirected and Weighted adjacency graph for the dataset. Only instances
            belonging to the same class must be connected. If not provided, will
            compute using `metric`, `p`, `radius_frac`

        Returns
        -------
        :obj:`np.ndarray`
            Complement of the hub score of every node.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           (Cited on page 9). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        """
        if adj_graph is None:
            adj_graph = cls._build_adjacency_graph(
                N=N,
                y=y,
                n_jobs=n_jobs,
                cls_inds=cls_inds,
                metric=metric,
                p=p,
                radius_frac=radius_frac,
                norm_dist_mat=norm_dist_mat,
            )

        hubs = 1.0 - np.asfarray(adj_graph.hub_score())

        return hubs
