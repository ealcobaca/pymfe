"""Module dedicated to extraction of Complexity Metafeatures."""

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
    def precompute_fx(cls,
                      y: np.ndarray,
                      cls_index_cp: bool = True,
                      cls_n_ex_cp: bool = True,
                      ovo_comb_cp: bool = True,
                      **kwargs
                      ) -> t.Dict[str, t.Any]:
        """Precompute some useful things to support feature-based measures.

        Parameters
        ----------
        y : :obj:`np.ndarray`, optional
            Target attribute from fitted data.

        cls_index_cp : bool, (default=True)
            True if you want to compute `cls_index`. It is the list of boolean
            vectors indicating the example of each class. The array indexes
            represent the classes.

        cls_n_ex_cp : bool, (default=True)
            True if you want to compute `cls_n_ex`. It is the number of
            examples in each class. The array indexes represent the classes.

        ovo_comb_cp : list
            True if you want to compute 'ovo_comb'. It is the List of all class
            OVO combination, i.e., [(0,1), (0,2) ...].

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
                - ``cls_index`` (:obj:`list`):  The list of boolean vectors
                  indicating the example of each class. The array indexes
                  represent the classes.
                  combination, i.e., [(0,1), (0,2) ...].
                - ``cls_n_ex`` (:obj:`np.ndarray`): The number of examples in
                  each class. The array indexes represent the classes.
        """
        prepcomp_vals = {}

        if y is not None and not {"classes", "class_freqs",
                                  "return_inverse"}.issubset(kwargs):
            prec = MFEGeneral.precompute_general_class(y)
            classes, y_idx, _ = (prec["classes"],
                                 prec["y_idx"], prec["class_freqs"])

        if (y is not None and
                "ovo_comb" not in kwargs and
                "cls_index" not in kwargs and
                "cls_n_ex" not in kwargs):
            if cls_index_cp or cls_n_ex_cp:
                cls_index = [np.equal(y_idx, i)
                             for i in range(classes.shape[0])]
            if cls_index_cp is True:
                prepcomp_vals["cls_index"] = cls_index
            if cls_n_ex_cp is True:
                cls_n_ex = np.array([np.sum(aux) for aux in cls_index])
                prepcomp_vals["cls_n_ex"] = cls_n_ex
            if ovo_comb_cp is True:
                ovo_comb = list(itertools.combinations(
                    range(classes.shape[0]), 2))
                prepcomp_vals["ovo_comb"] = ovo_comb
        return prepcomp_vals

    @classmethod
    def precompute_pca_tx(cls,
                          N: np.ndarray,
                          tx_n_components: float = 0.95,
                          **kwargs
                          ) -> t.Dict[str, t.Any]:
        """Precompute PCA to support dimensionality measures.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        **kwargs
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
        :obj:`dict`
            With following precomputed items:
                - ``m`` (:obj:`int`): Number of features.
                - ``m_`` (:obj:`int`):  Number of features after PCA with 0.95.
                - ``n`` (:obj:`int`): Number of examples.
        """
        prepcomp_vals = {}

        if (N is not None and
                "m" not in kwargs and
                "m_" not in kwargs and
                "n" not in kwargs):

            pca = PCA(n_components=tx_n_components)
            pca.fit(N)

            m_ = pca.explained_variance_ratio_.shape[0]
            m = N.shape[1]
            n = N.shape[0]

            prepcomp_vals["m_"] = m_
            prepcomp_vals["m"] = m
            prepcomp_vals["n"] = n

        return prepcomp_vals

    @staticmethod
    def _minmax(N: np.ndarray,
                class1: np.ndarray,
                class2: np.ndarray
                ) -> np.ndarray:
        """ This function computes the minimum of the maximum values per class
        for all features. The index i indicate the minmax of feature i.
        """

        min_cls = np.zeros((2, N.shape[1]))
        min_cls[0, :] = np.max(N[class1], axis=0)
        min_cls[1, :] = np.max(N[class2], axis=0)
        aux = np.min(min_cls, axis=0)
        return aux

    @staticmethod
    def _maxmin(N: np.ndarray,
                class1: np.ndarray,
                class2: np.ndarray
                ) -> np.ndarray:
        """ This function computes the maximum of the minimum values per class
        for all features. The index i indicate the maxmin of feature i.
        """

        max_cls = np.zeros((2, N.shape[1]))
        max_cls[0, :] = np.min(N[class1], axis=0)
        max_cls[1, :] = np.min(N[class2], axis=0)
        aux = np.max(max_cls, axis=0)
        return aux

    @staticmethod
    def _compute_f3(N_: np.ndarray,
                    minmax_: np.ndarray,
                    maxmin_: np.ndarray
                    ) -> np.ndarray:
        """ This function computes the F3 complexit measure given
        minmax and maxmin.
        """

        # True if the example is in the overlapping region
        # Should be > and < instead of >= and <= ?
        overlapped_region_by_feature = np.logical_and(
            N_ >= maxmin_, N_ <= minmax_)

        n_fi = np.sum(overlapped_region_by_feature, axis=0)
        idx_min = np.argmin(n_fi)

        return idx_min, n_fi, overlapped_region_by_feature

    @classmethod
    def ft_f3(cls,
              N: np.ndarray,
              ovo_comb: np.ndarray,
              cls_index: np.ndarray,
              cls_n_ex: np.ndarray,
              ) -> np.ndarray:
        """Computes the maximum individual feature efficiency measure for each
        feature.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        ovo_comb : :obj: `np.ndarray`
            List of all class OVO combination, i.e., [(0,1), (0,2) ...].

        cls_index : :obj:`list`
            The list of boolean vectors indicating the example of each class.
            The array indexes represent the classes combination, i.e.,
            [(0,1), (0,2) ...].

        cls_n_ex : :obj:`np.ndarray`
            The number of examples in each class. The array indexes represent
            the classes.

        Returns
        -------
        :obj:`np.ndarray`
            An array with the maximum individual feature efficiency measure for
            each feature.
        """
        f3 = []
        for idx1, idx2 in ovo_comb:
            idx_min, n_fi, _ = cls._compute_f3(
                N,
                cls._minmax(N, cls_index[idx1], cls_index[idx2]),
                cls._maxmin(N, cls_index[idx1], cls_index[idx2])
            )
            f3.append(n_fi[idx_min] / (cls_n_ex[idx1] + cls_n_ex[idx2]))

        # The measure is computed in the literature using the mean. However,
        # it is formulated here as a meta-feature. Therefore,
        # the post-processing should be used to get the mean and other measures
        # as well.
        # return np.mean(f3)

        return f3

    @classmethod
    def ft_f4(cls,
              N: np.ndarray,
              ovo_comb,
              cls_index,
              cls_n_ex,
              ) -> np.ndarray:
        """Computes the collective feature efficiency measure  for each
        feature.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        ovo_comb : :obj: `np.ndarray`
            List of all class OVO combination, i.e., [(0,1), (0,2) ...].

        cls_index : :obj:`list`
            The list of boolean vectors indicating the example of each class.
            The array indexes represent the classes combination, i.e.,
            [(0,1), (0,2) ...].

        cls_n_ex : :obj:`np.ndarray`
            The number of examples in each class. The array indexes represent
            the classes.

        Returns
        -------
        :obj:`np.ndarray`
            An array with the collective feature efficiency measure for each
            feature.
        """
        f4 = []
        for idx1, idx2 in ovo_comb:
            aux = 0

            y_class1 = cls_index[idx1]
            y_class2 = cls_index[idx2]
            sub_set = np.logical_or(y_class1, y_class2)
            y_class1 = y_class1[sub_set]
            y_class2 = y_class2[sub_set]
            N_ = N[sub_set, :]
            # N_ = N[np.logical_or(y_class1, y_class2),:]

            while N_.shape[1] > 0 and N_.shape[0] > 0:
                # True if the example is in the overlapping region
                idx_min, _, overlapped_region_by_feature = cls._compute_f3(
                    N_,
                    cls._minmax(N_, y_class1, y_class2),
                    cls._maxmin(N_, y_class1, y_class2)
                )

                # boolean that if True, this example is in the overlapping
                # region
                overlapped_region = overlapped_region_by_feature[:, idx_min]

                # removing the non overlapped features
                N_ = N_[overlapped_region, :]
                y_class1 = y_class1[overlapped_region]
                y_class2 = y_class2[overlapped_region]

                if N_.shape[0] > 0:
                    aux = N_.shape[0]
                else:
                    aux = 0

                # removing the most efficient feature
                N_ = np.delete(N_, idx_min, axis=1)

            f4.append(aux/(cls_n_ex[idx1] + cls_n_ex[idx2]))

        # The measure is computed in the literature using the mean. However,
        # it is formulated here as a meta-feature. Therefore,
        # the post-processing should be used to get the mean and other measures
        # as well.
        # return np.mean(f4)

        return f4

    @classmethod
    def ft_l2(cls,
              N: np.ndarray,
              ovo_comb: np.ndarray,
              cls_index: np.ndarray
              ) -> np.ndarray:
        """Computes the  error rate of linear classifier measure  for each
        OVO subset.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        ovo_comb : :obj: `np.ndarray`
            List of all class OVO combination, i.e., [(0,1), (0,2) ...].

        cls_index : :obj:`list`
            The list of boolean vectors indicating the example of each class.
            The array indexes represent the classes combination, i.e.,
            [(0,1), (0,2) ...].

        Returns
        -------
        :obj:`np.ndarray`
            An array with the collective  error rate of linear classifier
            measure for each OVO subset.
        """
        l2 = []
        for idx1, idx2 in ovo_comb:

            y_ = np.logical_or(cls_index[idx1], cls_index[idx2])
            N_ = N[y_, :]
            y_ = cls_index[idx1][y_]

            zscore = StandardScaler()
            svc = SVC(kernel='linear', C=1.0, tol=10e-3, max_iter=10e4)
            pip = Pipeline([('zscore', zscore), ('svc', svc)])
            pip.fit(N_, y_)
            y_pred = pip.predict(N_)
            error = 1 - accuracy_score(y_, y_pred)

            l2.append(error)

        # The measure is computed in the literature using the mean. However,
        # it is formulated here as a meta-feature. Therefore,
        # the post-processing should be used to get the mean and other measures
        # as well.
        # return np.mean(l2)

        return l2

    @classmethod
    def ft_n1(cls,
              N: np.ndarray,
              y: np.ndarray,
              metric: str = "euclidean"
              ) -> float:
        """Computes the fraction of borderline points measure.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        ovo_comb : :obj: `np.ndarray`
            List of all class OVO combination, i.e., [(0,1), (0,2) ...].

        cls_index : :obj:`list`
            The list of boolean vectors indicating the example of each class.
            The array indexes represent the classes combination, i.e.,
            [(0,1), (0,2) ...].

        Returns
        -------
        float
            An float with the fraction of borderline points measure.
        """
        # 0-1 scaler
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(N)
        N_ = scaler.transform(N)

        # compute the distance matrix and the minimum spanning tree.
        dist_m = np.triu(distance.cdist(N_, N_, metric), k=1)
        mst = minimum_spanning_tree(dist_m)
        node_i, node_j = np.where(mst.toarray() > 0)

        # which edges have nodes with different class
        which_have_diff_cls = y[node_i] != y[node_j]

        # I have doubts on how to compute it
        # 1) number of edges
        # aux = np.sum(which_have_diff_cls)

        # 2) number of different vertices connected
        aux = np.unique(np.concatenate([
            node_i[which_have_diff_cls],
            node_j[which_have_diff_cls]
        ])).shape[0]

        return float(aux/N.shape[0])

    @classmethod
    def ft_n4(cls,
              N: np.ndarray,
              y: np.ndarray,
              cls_index: np.ndarray,
              metric_n4: str = 'minkowski',
              p_n4: int = 2,
              n_neighbors_n4: int = 1
              ) -> float:
        """Computes the non-linearity of the nearest neighbor Classifier
        measure.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`, optional
            Target attribute from fitted data.

        cls_index : :obj:`list`
            The list of boolean vectors indicating the example of each class.
            The array indexes represent the classes combination, i.e.,
            [(0,1), (0,2) ...].

        metric_n4 : str, optional (default='minkowski')
            The distance metric used in the internal kNN classifier. See the
            documentation of the DistanceMetric class on sklearn for a list of
            available metrics.

        p_n4 : int, optional (default = 2)
            Power parameter for the Minkowski metric. When p = 1, this is
            equivalent to using manhattan_distance (l1), and
            euclidean_distance (l2) for p = 2. For arbitrary p,
            minkowski_distance (l_p) is used. Please, check the
            KNeighborsClassifier documentation on sklearn for more information.

        Returns
        -------
        float
            An float with the fraction of borderline points measure.
        """
        interp_N = []
        interp_y = []

        # 0-1 scaler
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(N)
        N = scaler.transform(N)

        for idx in cls_index:
            N_ = N[idx]

            A = np.random.choice(N_.shape[0], N_.shape[0])
            A = N_[A]
            B = np.random.choice(N_.shape[0], N_.shape[0])
            B = N_[B]
            delta = np.random.ranf(N_.shape)

            interp_N_ = A + ((B - A) * delta)
            interp_y_ = y[idx]

            interp_N.append(interp_N_)
            interp_y.append(interp_y_)

        # join the datasets
        N_test = np.concatenate(interp_N)
        y_test = np.concatenate(interp_y)

        knn = KNeighborsClassifier(
            n_neighbors=n_neighbors_n4, p=p_n4, metric=metric_n4).fit(N, y)
        y_pred = knn.predict(N_test)
        error = 1 - accuracy_score(y_test, y_pred)

        return float(error)

    @classmethod
    def ft_c1(cls,
              y: np.array,
              cls_n_ex: np.ndarray = None,
              ) -> float:
        """Computes the entropy of class proportions measure.

        Parameters
        ----------
        cls_n_ex : :obj:`np.ndarray`
            The number of examples in each class. The array indexes represent
            the classes.

        Returns
        -------
        :obj:`np.ndarray`
            An float with the entropy of class proportions measure.
        """
        if cls_n_ex is None:
            dict_precomputed = MFEComplexity.precompute_fx(
                y=y, cls_index_cp=False, cls_n_ex_cp=True,
                ovo_comb_cp=False)
            cls_n_ex = dict_precomputed["cls_n_ex"]

        nc = cls_n_ex.shape[0]
        pc_i = cls_n_ex / np.sum(cls_n_ex)
        c1 = -(1.0/np.log(nc)) * np.sum(pc_i*np.log(pc_i))

        # Shuldn't C1 be 1-C1? to match with C2?
        return float(c1)

    @classmethod
    def ft_c2(cls,
              cls_n_ex: np.ndarray
              ) -> float:
        """Computes the imbalance ratio measure.

        Parameters
        ----------
        cls_n_ex : :obj:`np.ndarray`
            The number of examples in each class. The array indexes represent
            the classes.

        Returns
        -------
        :obj:`np.ndarray`
            An float with the imbalance ratio measure.
        """
        n = np.sum(cls_n_ex)
        nc = cls_n_ex.shape[0]
        nc_i = cls_n_ex
        IR = ((nc-1) / nc) * np.sum(nc_i / (n - (nc_i)))
        c2 = 1 - (1/IR)

        return c2

    @classmethod
    def ft_t2(cls,
              m: int,
              n: int
              ) -> float:
        """Computes the average number of features per dimension measure.

        Parameters
        ----------
        m : int
            Number of features.
        n : int
            Number of examples.

        Returns
        -------
        :obj:`float`
            An float with the average number of features per dimension measure.
        """
        return m/n

    @classmethod
    def ft_t3(cls,
              m_: int,
              n: int
              ) -> float:
        """Computes the average number of PCA dimensions per points measure.

        Parameters
        ----------
        m_ : int
            Number of features after PCA with 0.95.
        n : int
            Number of examples.

        Returns
        -------
        :obj:`float`
            An float with the average number of PCA dimensions per points
            measure.
        """
        return m_/n

    @classmethod
    def ft_t4(cls,
              m: int,
              m_: int
              ) -> float:
        """Computes the ratio of the PCA dimension to the original dimension
        measure.

        Parameters
        ----------
        m : int
            Number of features.
        m_ : int
            Number of features after PCA with 0.95.

        Returns
        -------
        :obj:`float`
            An float with the ratio of the PCA dimension to the original
            dimension measure.
        """
        return m_/m
