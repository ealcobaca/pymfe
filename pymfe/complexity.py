"""Module dedicated to extraction of complexity metafeatures."""

import typing as t
import itertools

import numpy as np
from scipy.spatial import distance
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from pymfe.general import MFEGeneral
from pymfe.clustering import MFEClustering


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
       value or a generic Sequence (preferably a :obj:`np.ndarray`) type with
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
    def precompute_fx(cls, y: np.ndarray, **kwargs) -> t.Dict[str, t.Any]:
        """Precompute some useful things to support feature-based measures.

        Parameters
        ----------
        y : :obj:`np.ndarray`, optional
            Target attribute from fitted data.

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
                  rows represents the distinct classes, and the instances
                  are represented by the columns.
                - ``classes`` (:obj:`np.ndarray`): distinct classes in the
                  fitted target attribute.
                - ``class_freqs`` (:obj:`np.ndarray`): The number of examples
                  in each class. The indices represent the classes.
        """
        precomp_vals = {}

        if (y is not None and not {"classes", "class_freqs"}.issubset(kwargs)):
            sub_dic = MFEGeneral.precompute_general_class(y)
            precomp_vals.update(sub_dic)

        classes = kwargs.get("classes", precomp_vals.get("classes"))

        if (y is not None
                and ("ovo_comb" not in kwargs or "cls_inds" not in kwargs)):
            cls_inds = MFEComplexity._calc_cls_inds(y, classes)
            precomp_vals["cls_inds"] = cls_inds

            ovo_comb = MFEComplexity._calc_ovo_comb(classes)
            precomp_vals["ovo_comb"] = ovo_comb

        return precomp_vals

    @classmethod
    def precompute_pca_tx(cls,
                          N: np.ndarray,
                          tx_n_components: float = 0.95,
                          random_state: t.Optional[int] = None,
                          **kwargs) -> t.Dict[str, int]:
        """Precompute PCA to support dimensionality measures.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        tx_n_components : float, optional
            Minimum fraction of the fitted data variance that must be
            explained by the kept principal components after the PCA analysis.
            Check the ``sklearn.decomposition.PCA`` documentation for more
            detailed information about this argument.

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
        precomp_vals = {}

        if N is not None and "num_attr_pca" not in kwargs:
            pca = PCA(n_components=tx_n_components, random_state=random_state)
            pca.fit(N)

            num_attr_pca = pca.explained_variance_ratio_.shape[0]

            precomp_vals["num_attr_pca"] = num_attr_pca

        return precomp_vals

    @staticmethod
    def _calc_cls_inds(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """Compute the ``cls_inds`` variable.

        The ``cls_inds`` variable is a boolean array which marks with
        True value whether the instance belongs to each class. Each
        distinct class is represented by a row, and each instance is
        represented by a column.
        """
        cls_inds = np.array([np.equal(y, cur_cls) for cur_cls in classes],
                            dtype=bool)

        return cls_inds

    @staticmethod
    def _calc_ovo_comb(classes: np.ndarray) -> t.List[t.Tuple]:
        """Compute the ``ovo_comb`` variable.

        The ``ovo_comb`` value is a array with all class OVO combination,
        i.e., all combinations of distinct class indices by pairs.
        """
        ovo_comb = itertools.combinations(np.arange(classes.size), 2)
        return np.asarray(list(ovo_comb), dtype=int)

    @staticmethod
    def _calc_minmax(N: np.ndarray, class1: np.ndarray,
                     class2: np.ndarray) -> np.ndarray:
        """Compute the minimum of the maximum values per class for all feat.

        The index i indicate the minmax of feature i.
        """
        minmax = np.min((np.max(N[class1, :], axis=0),
                         np.max(N[class2, :], axis=0)), axis=0)
        return minmax

    @staticmethod
    def _calc_maxmin(N: np.ndarray, class1: np.ndarray,
                     class2: np.ndarray) -> np.ndarray:
        """Compute the maximum of the minimum values per class for all feat.

        The index i indicate the maxmin of the ith feature.
        """
        maxmin = np.max((np.min(N[class1, :], axis=0),
                         np.min(N[class2, :], axis=0)), axis=0)
        return maxmin

    @staticmethod
    def _calc_overlap(
            N: np.ndarray,
            minmax: np.ndarray,
            maxmin: np.ndarray,
    ) -> np.ndarray:
        """Compute the F3 complexit measure given minmax and maxmin."""
        # True if the example is in the overlapping region
        # Should be > and < instead of >= and <= ?
        # TODO: the MFE (R version) implements the non-overlapping region
        # as (N < maxmin || N > minmax) which, consequently, the contra-
        # positive version implies that the overlapping region is
        # (N >= maxmin && N <= minmax), just like is implemented here.
        # The question is: is the MFE (R version) behaviour correct?
        feat_overlapped_region = np.logical_and(N >= maxmin, N <= minmax)

        feat_overlap_num = np.sum(feat_overlapped_region, axis=0)
        ind_less_overlap = np.argmin(feat_overlap_num)

        return ind_less_overlap, feat_overlap_num, feat_overlapped_region

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

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Fitted target attribute.

        ovo_comb : :obj:`np.ndarray`, optional
            List of all class OVO combination, i.e., all combinations of
            distinct class indices by pairs ([(0, 1), (0, 2) ...].)

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows represents each distinct class, and the columns
            represents the instances.

        class_freqs : :obj:`np.ndarray`, optional
            The number of examples in each class. The indices represent
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
           Page 6.
        """
        if ovo_comb is None or cls_inds is None or class_freqs is None:
            sub_dic = MFEComplexity.precompute_fx(y=y)
            ovo_comb = sub_dic["ovo_comb"]
            cls_inds = sub_dic["cls_inds"]
            class_freqs = sub_dic["class_freqs"]

        f3 = np.zeros(ovo_comb.shape[0], dtype=float)

        for ind, (idx1, idx2) in enumerate(ovo_comb):
            ind_less_overlap, feat_overlap_num, _ = cls._calc_overlap(
                N=N,
                minmax=cls._calc_minmax(N, cls_inds[idx1], cls_inds[idx2]),
                maxmin=cls._calc_maxmin(N, cls_inds[idx1], cls_inds[idx2]))

            f3[ind] = (feat_overlap_num[ind_less_overlap] /
                       (class_freqs[idx1] + class_freqs[idx2]))

        # The measure is computed in the literature using the mean. However,
        # it is formulated here as a meta-feature. Therefore,
        # the post-processing should be used to get the mean and other measures
        # as well.
        # return np.mean(f3)

        return np.asarray(f3)

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

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Fitted target attribute.

        ovo_comb : :obj:`np.ndarray`, optional
            List of all class OVO combination, i.e., all combinations of
            distinct class indices by pairs ([(0, 1), (0, 2) ...].)

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows represents each distinct class, and the columns
            represents the instances.

        class_freqs : :obj:`np.ndarray`, optional
            The number of examples in each class. The indices represent
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
           Page 7.
        """
        if ovo_comb is None or cls_inds is None or class_freqs is None:
            sub_dic = MFEComplexity.precompute_fx(y=y)
            ovo_comb = sub_dic["ovo_comb"]
            cls_inds = sub_dic["cls_inds"]
            class_freqs = sub_dic["class_freqs"]

        f4 = np.zeros(ovo_comb.shape[0], dtype=float)

        for ind, (idx1, idx2) in enumerate(ovo_comb):
            y_class1 = cls_inds[idx1]
            y_class2 = cls_inds[idx2]
            cls_subset_intersec = np.logical_or(y_class1, y_class2)
            y_class1 = y_class1[cls_subset_intersec]
            y_class2 = y_class2[cls_subset_intersec]
            N_subset = N[cls_subset_intersec, :]

            while N_subset.size > 0:
                # True if the example is in the overlapping region
                ind_less_overlap, _, feat_overlapped_region = (
                    cls._calc_overlap(
                        N=N_subset,
                        minmax=cls._calc_minmax(N_subset, y_class1, y_class2),
                        maxmin=cls._calc_maxmin(N_subset, y_class1, y_class2)))

                # boolean that if True, this example is in the overlapping
                # region
                overlapped_region = feat_overlapped_region[:, ind_less_overlap]

                # removing the non overlapped features
                N_subset = N_subset[overlapped_region, :]
                y_class1 = y_class1[overlapped_region]
                y_class2 = y_class2[overlapped_region]

                # removing the most efficient feature
                N_subset = np.delete(N_subset, ind_less_overlap, axis=1)

            subset_size = N_subset.shape[0]

            f4[ind] = subset_size / (class_freqs[idx1] + class_freqs[idx2])

        # The measure is computed in the literature using the mean. However,
        # it is formulated here as a meta-feature. Therefore,
        # the post-processing should be used to get the mean and other measures
        # as well.
        # return np.mean(f4)

        return np.asarray(f4)

    @classmethod
    def ft_l2(cls,
              N: np.ndarray,
              y: np.ndarray,
              ovo_comb: t.Optional[np.ndarray] = None,
              cls_inds: t.Optional[np.ndarray] = None,
              max_iter: t.Union[int, float] = 1e5) -> np.ndarray:
        """Compute the OVO subsets error rate of linear classifier.

        The linear model used is induced by the Support Vector
        Machine algorithm.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Fitted target attribute.

        ovo_comb : :obj:`np.ndarray`, optional
            List of all class OVO combination, i.e., all combinations of
            distinct class indices by pairs ([(0, 1), (0, 2) ...].)

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows represents each distinct class, and the columns
            represents the instances.

        max_iter : float or int, optional
            Maximum number of iterations allowed for the support vector
            machine model convergence. This parameter can receive float
            numbers to be compatible with the Python scientific notation
            data type.

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
           Page 9.
        """
        if ovo_comb is None or cls_inds is None:
            sub_dic = MFEComplexity.precompute_fx(y=y)
            ovo_comb = sub_dic["ovo_comb"]
            cls_inds = sub_dic["cls_inds"]

        l2 = np.zeros(ovo_comb.shape[0], dtype=float)

        zscore = StandardScaler()
        svc = SVC(kernel="linear", C=1.0, tol=10e-3, max_iter=int(max_iter))
        pip = Pipeline([("zscore", zscore), ("svc", svc)])

        for ind, (idx1, idx2) in enumerate(ovo_comb):
            cls_intersec = np.logical_or(cls_inds[idx1], cls_inds[idx2])

            N_subset = N[cls_intersec, :]
            y_subset = cls_inds[idx1, cls_intersec]

            pip.fit(N_subset, y_subset)
            y_pred = pip.predict(N_subset)
            error = 1 - accuracy_score(y_subset, y_pred)

            l2[ind] = error

        # The measure is computed in the literature using the mean. However,
        # it is formulated here as a meta-feature. Therefore,
        # the post-processing should be used to get the mean and other measures
        # as well.
        # return np.mean(l2)

        return np.asarray(l2)

    @classmethod
    def ft_n1(cls, N: np.ndarray, y: np.ndarray,
              metric: str = "euclidean") -> float:
        """Compute the fraction of borderline points.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Fitted target attribute.

        metric : str, optional
            Metric used to calculate the distances between the instances.
            Check the ``scipy.spatial.distance.cdist`` documentation to
            get a list of all available metrics.

        Returns
        -------
        float
            Fraction of borderline points.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           Page 9-10.
        """
        # 0-1 feature scaling
        N = MinMaxScaler(feature_range=(0, 1)).fit_transform(N)

        # Compute the distance matrix and the minimum spanning tree.
        dist_mat = np.triu(distance.cdist(N, N, metric=metric), k=1)
        mst = minimum_spanning_tree(dist_mat, overwrite=True)
        node_id_i, node_id_j = np.nonzero(mst)

        # Which edges have nodes with different class
        which_have_diff_cls = y[node_id_i] != y[node_id_j]

        # Number of vertices connected with different classes
        borderline_inst_num = np.unique(
            np.concatenate([
                node_id_i[which_have_diff_cls],
                node_id_j[which_have_diff_cls],
            ])).size

        inst_num = N.shape[0]

        return borderline_inst_num / inst_num

    @classmethod
    def ft_n4(cls,
              N: np.ndarray,
              y: np.ndarray,
              cls_inds: t.Optional[np.ndarray] = None,
              metric_n4: str = "minkowski",
              p_n4: int = 2,
              n_neighbors_n4: int = 1,
              random_state: t.Optional[int] = None) -> float:
        """Compute the non-linearity of the NN Classifier.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows represents each distinct class, and the columns
            represents the instances.

        metric_n4 : str, optional (default = "minkowski")
            The distance metric used in the internal kNN classifier. See the
            documentation of the ``sklearn.neighbors.DistanceMetric`` class
            for a list of available metrics.

        p_n4 : int, optional (default = 2)
            Power parameter for the Minkowski metric. When p = 1, this is
            equivalent to using manhattan_distance (l1), and
            euclidean_distance (l2) for p = 2. For arbitrary p,
            minkowski_distance (l_p) is used. Please, check the
            ``sklearn.neighbors.KNeighborsClassifier`` documentation for
            more information.

        random_state : int, optional
            If given, set the random seed before computing the randomized
            data interpolation.

        Returns
        -------
        float
            Estimated non-linearity of the NN classifier.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           Page 11.
        """
        if cls_inds is None:
            classes = np.unique(y)
            cls_inds = MFEComplexity._calc_cls_inds(y, classes)

        # 0-1 feature scaling
        N = MinMaxScaler(feature_range=(0, 1)).fit_transform(N)

        if random_state is not None:
            np.random.seed(random_state)

        N_test = np.zeros(N.shape, dtype=N.dtype)
        y_test = np.zeros(y.shape, dtype=y.dtype)

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
            N_test[ind_cur:ind_next, :] = N_subset_interp
            y_test[ind_cur:ind_next] = y[inds_cur_cls]
            ind_cur = ind_next

        knn = KNeighborsClassifier(
            n_neighbors=n_neighbors_n4, p=p_n4, metric=metric_n4).fit(N, y)

        y_pred = knn.predict(N_test)

        # TODO: The MFE (R version) returns an boolean array marking all
        # misclassifications, in order to be summarized by the framework.
        # Should we adopt the same strategy?
        # return np.not_equal(y_test, y_pred)
        error = 1 - accuracy_score(y_test, y_pred)

        return error

    @classmethod
    def ft_c1(
            cls,
            y: np.array,
            class_freqs: t.Optional[np.ndarray] = None,
    ) -> float:
        """Compute the entropy of class proportions.

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        class_freqs : :obj:`np.ndarray`, optional
            The number of examples in each class. The indices represent
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
           Page 15.
        """
        if class_freqs is None:
            _, class_freqs = np.unique(y, return_counts=True)

        num_class = class_freqs.size

        # Note: calling 'ft_nre' just to make explicity the link
        # between this metafeature and 'C1'.
        c1 = MFEClustering.ft_nre(
            y=y, class_freqs=class_freqs) / np.log(num_class)

        return c1

    @classmethod
    def ft_c2(cls, y: np.ndarray,
              class_freqs: t.Optional[np.ndarray] = None) -> float:
        """Compute the imbalance ratio.

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        class_freqs : :obj:`np.ndarray`, optional
            The number of examples in each class. The indices represent
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
           Page 16.
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
    def ft_t2(cls, X: np.ndarray) -> float:
        """Compute the average number of features per dimension.

        Parameters
        ----------
        X : :obj:`np.ndarray`
            Original attributes from fitted data.

        Returns
        -------
        float
            Average number of features per dimension.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           Page 15.
        """
        # Note: calling 'ft_attr_to_inst' just to make explicity the link
        # between this metafeature and 'T2'.
        return MFEGeneral.ft_attr_to_inst(X=X)

    @classmethod
    def ft_t3(
            cls,
            N: np.ndarray,
            num_attr_pca: t.Optional[int] = None,
            random_state: t.Optional[int] = None,
    ) -> float:
        """Compute the average number of PCA dimensions per points.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

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
           Page 15.
        """
        if num_attr_pca is None:
            sub_dic = MFEComplexity.precompute_pca_tx(
                N=N, random_state=random_state)
            num_attr_pca = sub_dic["num_attr_pca"]

        num_inst = N.shape[0]

        return num_attr_pca / num_inst

    @classmethod
    def ft_t4(cls,
              N: np.ndarray,
              num_attr_pca: t.Optional[int] = None,
              random_state: t.Optional[int] = None) -> float:
        """Compute the ratio of the PCA dimension to the original dimension.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

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
           Page 15.
        """
        if num_attr_pca is None:
            sub_dic = MFEComplexity.precompute_pca_tx(
                N=N, random_state=random_state)
            num_attr_pca = sub_dic["num_attr_pca"]

        num_attr = N.shape[1]

        return num_attr_pca / num_attr
