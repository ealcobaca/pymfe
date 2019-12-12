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
                - ``ovo_comb`` (:obj:`list`): List of all class OVO
                  combination, i.e., [(0,1), (0,2) ...].
                - ``cls_inds`` (:obj:`list`): The list of boolean vectors
                  indicating the example of each class. The array indexes
                  represent the classes.
                  combination, i.e., [(0,1), (0,2) ...].
                - ``class_freqs`` (:obj:`np.ndarray`): The number of examples
                  in each class. The array indexes represent the classes.
        """
        precomp_vals = {}

        if (y is not None
                and not {"classes", "y_idx", "class_freqs"}.issubset(kwargs)):
            sub_dic = MFEGeneral.precompute_general_class(y)
            precomp_vals.update(sub_dic)

        classes = precomp_vals["classes"]
        y_idx = precomp_vals["y_idx"]

        if (y is not None
                and ("ovo_comb" not in kwargs or "cls_inds" not in kwargs)):
            cls_inds = MFEComplexity._compute_cls_inds(y_idx, classes)
            precomp_vals["cls_inds"] = cls_inds

            ovo_comb = MFEComplexity._compute_ovo_comb(classes)
            precomp_vals["ovo_comb"] = ovo_comb

        return precomp_vals

    @classmethod
    def precompute_pca_tx(cls,
                          N: np.ndarray,
                          tx_n_components: float = 0.95,
                          **kwargs) -> t.Dict[str, int]:
        """Precompute PCA to support dimensionality measures.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        tx_n_components : :obj:`float`, optional
            Number of principal components to keep in the PCA.

        **kwargs
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
        :obj:`dict`
            With following precomputed items:
                - ``num_attr_pca`` (:obj:`int`): Number of features after PCA
                  with at least 0.95 fraction of variance is explained by the
                  selected components.
        """
        precomp_vals = {}

        if N is not None and "num_attr_pca" not in kwargs:
            pca = PCA(n_components=tx_n_components)
            pca.fit(N)

            num_attr_pca = pca.explained_variance_ratio_.shape[0]

            precomp_vals["num_attr_pca"] = num_attr_pca

        return precomp_vals

    @staticmethod
    def _compute_cls_inds(y_idx: np.ndarray,
                          classes: np.ndarray) -> t.List[np.ndarray]:
        """Computes the ``cls_inds`` variable."""
        return [np.equal(y_idx, i) for i in np.arange(classes.size)]

    @staticmethod
    def _compute_ovo_comb(classes: np.ndarray) -> t.List[t.Tuple]:
        """Computes the ``ovo_comb`` variable."""
        ovo_comb = itertools.combinations(np.arange(classes.size), 2)
        return np.asarray(list(ovo_comb), dtype=int)

    @staticmethod
    def _minmax(N: np.ndarray, class1: np.ndarray,
                class2: np.ndarray) -> np.ndarray:
        """Computes the minimum of the maximum values per class for all feat.

        The index i indicate the minmax of feature i.
        """
        min_cls = np.zeros((2, N.shape[1]))
        min_cls[0, :] = np.max(N[class1, :], axis=0)
        min_cls[1, :] = np.max(N[class2, :], axis=0)
        return np.min(min_cls, axis=0)

    @staticmethod
    def _maxmin(N: np.ndarray, class1: np.ndarray,
                class2: np.ndarray) -> np.ndarray:
        """Computes the maximum of the minimum values per class for all feat.

        The index i indicate the maxmin of feature i.
        """
        max_cls = np.zeros((2, N.shape[1]))
        max_cls[0, :] = np.min(N[class1, :], axis=0)
        max_cls[1, :] = np.min(N[class2, :], axis=0)
        return np.max(max_cls, axis=0)

    @staticmethod
    def _compute_f3(N: np.ndarray, minmax: np.ndarray,
                    maxmin: np.ndarray) -> np.ndarray:
        """Compute the F3 complexit measure given minmax and maxmin."""
        # True if the example is in the overlapping region
        # Should be > and < instead of >= and <= ?
        overlapped_region_by_feature = np.logical_and(N >= maxmin, N <= minmax)

        n_fi = np.sum(overlapped_region_by_feature, axis=0)
        idx_min = np.argmin(n_fi)

        return idx_min, n_fi, overlapped_region_by_feature

    @classmethod
    def ft_f3(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            ovo_comb: np.ndarray = None,
            cls_inds: np.ndarray = None,
            class_freqs: np.ndarray = None,
    ) -> np.ndarray:
        """Computes each feature maximum individual efficiency.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        ovo_comb : :obj:`np.ndarray`
            List of all class OVO combination, i.e., [(0,1), (0,2) ...].

        cls_inds : :obj:`list`
            The list of boolean vectors indicating the example of each class.
            The array indexes represent the classes combination, i.e.,
            [(0,1), (0,2) ...].

        class_freqs : :obj:`np.ndarray`
            The number of examples in each class. The array indexes represent
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
            idx_min, n_fi, _ = cls._compute_f3(
                N=N,
                minmax=cls._minmax(N, cls_inds[idx1], cls_inds[idx2]),
                maxmin=cls._maxmin(N, cls_inds[idx1], cls_inds[idx2]))

            f3[ind] = n_fi[idx_min] / (class_freqs[idx1] + class_freqs[idx2])

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
            ovo_comb: np.ndarray = None,
            cls_inds: np.ndarray = None,
            class_freqs: np.ndarray = None,
    ) -> np.ndarray:
        """Computes the features collective feature efficiency.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        ovo_comb : :obj:`np.ndarray`
            List of all class OVO combination, i.e., [(0,1), (0,2) ...].

        cls_inds : :obj:`list`
            The list of boolean vectors indicating the example of each class.
            The array indexes represent the classes combination, i.e.,
            [(0,1), (0,2) ...].

        class_freqs : :obj:`np.ndarray`
            The number of examples in each class. The array indexes represent
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
                idx_min, _, overlapped_region_by_feature = cls._compute_f3(
                    N=N_subset,
                    minmax=cls._minmax(N_subset, y_class1, y_class2),
                    maxmin=cls._maxmin(N_subset, y_class1, y_class2))

                # boolean that if True, this example is in the overlapping
                # region
                overlapped_region = overlapped_region_by_feature[:, idx_min]

                # removing the non overlapped features
                N_subset = N_subset[overlapped_region, :]
                y_class1 = y_class1[overlapped_region]
                y_class2 = y_class2[overlapped_region]

                # removing the most efficient feature
                N_subset = np.delete(N_subset, idx_min, axis=1)

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
              ovo_comb: np.ndarray = None,
              cls_inds: np.ndarray = None,
              max_iter: t.Union[int, float] = 1e5) -> np.ndarray:
        """Computes the OVO subsets error rate of linear classifier.

        The linear model used is induced by the Support Vector
        Machine algorithm.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        ovo_comb : :obj:`np.ndarray`, optional
            List of all class OVO combination, i.e., [(0,1), (0,2) ...].

        cls_inds : list, optional
            The list of boolean vectors indicating the example of each class.
            The array indexes represent the classes combination, i.e.,
            [(0,1), (0,2) ...].

        max_iter : float or int, optional
            Maximum number of iterations allowed for the support vector
            machine model convergence. This parameter can receive float
            numbers to be compatible with Python scientific notation
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
            y_subset = cls_inds[idx1][cls_intersec]

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
        """Computes the fraction of borderline points measure.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        ovo_comb : :obj:`np.ndarray`
            List of all class OVO combination, i.e., [(0,1), (0,2) ...].

        cls_inds : :obj:`list`
            The list of boolean vectors indicating the example of each class.
            The array indexes represent the classes combination, i.e.,
            [(0,1), (0,2) ...].

        Returns
        -------
        :obj:`float`
            Fraction of borderline points measure.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           Page 9-10.
        """
        # 0-1 scaling
        N = MinMaxScaler(feature_range=(0, 1)).fit_transform(N)

        # Compute the distance matrix and the minimum spanning tree.
        dist_m = np.triu(distance.cdist(N, N, metric), k=1)
        mst = minimum_spanning_tree(dist_m)
        node_id_i, node_id_j = np.nonzero(mst)

        # Which edges have nodes with different class
        which_have_diff_cls = y[node_id_i] != y[node_id_j]

        # Number of vertices connected with different classes
        borderline_inst_num = np.unique(
            np.concatenate([
                node_id_i[which_have_diff_cls], node_id_j[which_have_diff_cls]
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
        """Computes the non-linearity of the NN Classifier.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        cls_inds : :obj:`list`, optional
            The list of boolean vectors indicating the example of each class.
            The array indexes represent the classes combination, i.e.,
            [(0,1), (0,2) ...].

        metric_n4 : :obj:`str`, optional (default = "minkowski")
            The distance metric used in the internal kNN classifier. See the
            documentation of the DistanceMetric class on sklearn for a list of
            available metrics.

        p_n4 : :obj:`int`, optional (default = 2)
            Power parameter for the Minkowski metric. When p = 1, this is
            equivalent to using manhattan_distance (l1), and
            euclidean_distance (l2) for p = 2. For arbitrary p,
            minkowski_distance (l_p) is used. Please, check the
            KNeighborsClassifier documentation on sklearn for more information.

        random_state : int, optional
            If given, set the random seed before computing any pseudo-random
            value.

        Returns
        -------
        :obj:`float`
            Estimated non-linearity of the NN classifier.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           Page 11.
        """
        if cls_inds is None:
            sub_dic = MFEGeneral.precompute_general_class(y)

            classes = sub_dic["classes"]
            y_idx = sub_dic["y_idx"]

            cls_inds = MFEComplexity._compute_cls_inds(y_idx, classes)

        # 0-1 scaling
        N = MinMaxScaler(feature_range=(0, 1)).fit_transform(N)

        if random_state is not None:
            np.random.seed(random_state)

        N_test = np.zeros(N.shape, dtype=N.dtype)
        y_test = np.zeros(y.shape, dtype=y.dtype)

        ind_cur = 0

        for inds_cur_cls in cls_inds:
            N_cur_class = N[inds_cur_cls, :]
            subset_size = N_cur_class.shape[0]

            # Currently it is allowed to a instance 'interpolate with itself'
            # holding the instance itself as the result
            A = N_cur_class[np.random.choice(subset_size, subset_size), :]
            B = N_cur_class[np.random.choice(subset_size, subset_size), :]

            rand_noise = np.random.ranf(N_cur_class.shape)

            ind_next = ind_cur + subset_size
            N_test[ind_cur:ind_next, :] = A + ((B - A) * rand_noise)
            y_test[ind_cur:ind_next] = y[inds_cur_cls]
            ind_cur = ind_next

        knn = KNeighborsClassifier(
            n_neighbors=n_neighbors_n4, p=p_n4, metric=metric_n4).fit(N, y)

        y_pred = knn.predict(N_test)

        error = 1 - accuracy_score(y_test, y_pred)

        return error

    @classmethod
    def ft_c1(
            cls,
            y: np.array,
            class_freqs: t.Optional[np.ndarray] = None,
    ) -> float:
        """Computes the entropy of class proportions measure.

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        class_freqs : :obj:`np.ndarray`, optional
            The number of examples in each class. The array indexes represent
            the classes.

        Returns
        -------
        :obj:`float`
            Entropy of class proportions measure.

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
        num_inst = y.size

        pc_i = class_freqs / num_inst
        c1 = -np.sum(pc_i * np.log(pc_i)) / np.log(num_class)

        # Shuldn't C1 be 1-C1? to match with C2?
        return c1

    @classmethod
    def ft_c2(cls, y: np.ndarray,
              class_freqs: t.Optional[np.ndarray] = None) -> float:
        """Compute the imbalance ratio measure.

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        class_freqs : :obj:`np.ndarray`, optional
            The number of examples in each class. The array indexes represent
            the classes.

        Returns
        -------
        :obj:`float`
            The imbalance ratio measure.

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

        IR = np.sum(class_freqs / (num_inst - class_freqs))
        IR *= (num_class - 1) / num_class

        c2 = 1 - (1 / IR)

        return c2

    @classmethod
    def ft_t2(cls, N: np.ndarray) -> float:
        """Compute the average number of features per dimension measure.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        Returns
        -------
        :obj:`float`
            Average number of features per dimension measure.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           Page 15.
        """
        num_inst, num_attr = N.shape
        return num_attr / num_inst

    @classmethod
    def ft_t3(
            cls,
            N: np.ndarray,
            num_attr_pca: t.Optional[int] = None,
    ) -> float:
        """Computes the average number of PCA dimensions per points measure.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        num_attr_pca : :obj:`int`, optional
            Number of features after PCA where a fraction of at least 0.95
            of the data variance is explained by the selected components.

        Returns
        -------
        :obj:`float`
            Average number of PCA dimensions per points measure.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           Page 15.
        """
        if num_attr_pca is None:
            sub_dic = MFEComplexity.precompute_pca_tx(N=N)
            num_attr_pca = sub_dic["num_attr_pca"]

        num_inst = N.shape[0]

        return num_attr_pca / num_inst

    @classmethod
    def ft_t4(cls, N: np.ndarray,
              num_attr_pca: t.Optional[int] = None) -> float:
        """Computes the ratio of the PCA dimension to the original dimension.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        num_attr_pca : :obj:`int`, optional
            Number of features after PCA where a fraction of at least 0.95
            of the data variance is explained by the selected components.

        Returns
        -------
        :obj:`float`
            An float with the ratio of the PCA dimension to the original
            dimension measure.

        References
        ----------
        .. [1] Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P.
           Souto, and Tin K. Ho. How Complex is your classification problem?
           A survey on measuring classification complexity (V2). (2019)
           Page 15.
        """
        if num_attr_pca is None:
            sub_dic = MFEComplexity.precompute_pca_tx(N=N)
            num_attr_pca = sub_dic["num_attr_pca"]

        num_attr = N.shape[1]

        return num_attr_pca / num_attr
