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


class MFEComplexity:
    """Keep methods for metafeatures of ``landmarking`` group.

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
                      N: np.ndarray,
                      y: np.ndarray,
                      **kwargs
                      ) -> t.Dict[str, t.Any]:

        prepcomp_vals = {}

        classes, idx_classes, y_idx, class_freqs = np.unique(
            y, return_index=True, return_inverse=True, return_counts=True)

        if (y is not None and
                "ovo_comb" not in kwargs and
                "cls_index" not in kwargs and
                "cls_amount" not in kwargs):
            n_classe = 2
            cls_index = [np.equal(y_idx, i)
                         for i in range(classes.shape[0])]
            cls_amount = np.array([np.sum(aux) for aux in cls_index])
            ovo_comb = list(itertools.combinations(range(classes.shape[0]), 2))
            prepcomp_vals["ovo_comb"] = ovo_comb
            prepcomp_vals["cls_index"] = cls_index
            prepcomp_vals["cls_amount"] = cls_amount
        return prepcomp_vals

    @staticmethod
    def _minmax(N: np.ndarray,
                class1:np.ndarray,
                class2:np.ndarray
                ) -> np.ndarray:
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

        # True if the example is in the overlapping region
        # Should be > and < instead of >= and <= ?
        overlapped_region_by_feature = np.logical_and(
            N_ >= maxmin_, N_ <= minmax_)

        n_fi = np.sum(overlapped_region_by_feature, axis=0)
        idx_min = np.argmin(n_fi)

        return idx_min, n_fi, overlapped_region_by_feature

    @classmethod
    def ft_F3(cls,
              N: np.ndarray,
              ovo_comb: np.ndarray,
              cls_index: np.ndarray,
              cls_amount: np.ndarray,
              ) -> np.ndarray:
        """Performance of a the best single decision tree node.

        Construct a single decision tree node model induced by the most
        informative attribute to establish the linear separability.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        Returns
        -------
        :obj:`np.ndarray`
            The performance of each fold.
        """
        f3 = []
        for idx1, idx2 in ovo_comb:
            idx_min, n_fi, _ = cls._compute_f3(
                N,
                cls._minmax(N, cls_index[idx1], cls_index[idx2]),
                cls._maxmin(N, cls_index[idx1], cls_index[idx2])
            )
            f3.append(n_fi[idx_min] / (cls_amount[idx1] + cls_amount[idx2]))

        return np.mean(f3)

    @classmethod
    def ft_F4(cls,
              N: np.ndarray,
              ovo_comb,
              cls_index,
              cls_amount,
              ) -> np.ndarray:
        """Performance of a the best single decision tree node.

        Construct a single decision tree node model induced by the most
        informative attribute to establish the linear separability.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        Returns
        -------
        :obj:`np.ndarray`
            The performance of each fold.
        """

        f4 = []
        for idx1, idx2 in ovo_comb:
            no_overlap = False
            aux = 0

            y_class1 = cls_index[idx1]
            y_class2 = cls_index[idx2]
            sub_set = np.logical_or(y_class1, y_class2)
            y_class1 = y_class1[sub_set]
            y_class2 = y_class2[sub_set]
            N_ = N[sub_set, :]
            # N_ = N[np.logical_or(y_class1, y_class2),:]

            for i in range(N.shape[1]):
                # True if the example is in the overlapping region
                idx_min, _, overlapped_region_by_feature = cls._compute_f3(
                    N_,
                    cls._minmax(N_, y_class1, y_class2),
                    cls._maxmin(N_, y_class1, y_class2)
                )

                # boolean that if True, this example is in the overlapping region
                overlapped_region = overlapped_region_by_feature[:, idx_min]

                # removing the non overlapped features
                N_ = N_[overlapped_region, :]
                y_class1 = y_class1[overlapped_region]
                y_class2 = y_class2[overlapped_region]

                if N_.shape[0] == 0:
                    aux = 0
                    break
                else:
                    aux = N_.shape[0]

                # removing the most efficient feature
                N_ = np.delete(N_, idx_min, axis=1)

            f4.append(aux/(cls_amount[idx1] + cls_amount[idx2]))

        return np.mean(f4)

    @classmethod
    def ft_L2(cls,
              N: np.ndarray,
              ovo_comb: np.ndarray,
              cls_index: np.ndarray) -> np.ndarray:

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

        return np.mean(l2)

    # @classmethod
    # def ft_N1(cls,
    #           N: np.ndarray,
    #           ovo_comb: np.ndarray,
    #           cls_index: np.ndarray,
    #           dist_measure: str ="euclidean"
    #           ) -> np.ndarray:
    #
    #     n1 = []
    #     for idx1, idx2 in ovo_comb:
    #
    #         y_ = np.logical_or(cls_index[idx1], cls_index[idx2])
    #         N_ = N[y_, :]
    #         y_ = cls_index[idx1][y_]
    #
    #         # 0-1 scaler
    #         scaler = MinMaxScaler(feature_range=(0, 1)).fit(N_)
    #         new_N_ = scaler.transform(N_)
    #
    #         # ###
    #         dist_m = np.triu(distance.cdist(new_N_, new_N_, dist_measure), k=1)
    #         mst = minimum_spanning_tree(dist_m)
    #         node_i, node_j = np.where(mst.toarray() > 0)
    #         # ###
    #         which_have_diff_cls = np.logical_xor(y_[node_i], y_[node_j])
    #
    #         # I have doubts on how to compute it
    #         # 1) number of edges
    #         # aux = np.sum(which_have_diff_cls)
    #
    #         # 2) number of different vertices connected
    #         aux = np.unique(np.concatenate([
    #             node_i[which_have_diff_cls],
    #             node_j[which_have_diff_cls]
    #         ])).shape[0]
    #
    #         n1.append(aux / N_.shape[0])
    #
    #     print(n1)
    #     return np.mean(n1)

    @classmethod
    def ft_N1(cls,
              N: np.ndarray,
              y: np.ndarray,
              metric: str ="euclidean"
              ) -> np.ndarray:

        # 0-1 scaler
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(N)
        N_ = scaler.transform(N)

        # ###
        dist_m = np.triu(distance.cdist(N_, N_, metric), k=1)
        mst = minimum_spanning_tree(dist_m)
        node_i, node_j = np.where(mst.toarray() > 0)
        # ###
        which_have_diff_cls = y[node_i] != y[node_j]

        # I have doubts on how to compute it
        # 1) number of edges
        # aux = np.sum(which_have_diff_cls)

        # 2) number of different vertices connected
        aux = np.unique(np.concatenate([
            node_i[which_have_diff_cls],
            node_j[which_have_diff_cls]
        ])).shape[0]

        return aux/N.shape[0]

    @classmethod
    def ft_N4(cls,
              N: np.ndarray,
              y: np.ndarray,
              cls_index: np.ndarray,
              metric: str ="euclidean",
              p=2,
              n_neighbors=1
              ) -> np.ndarray:

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
            n_neighbors=n_neighbors, p=p, metric=metric).fit(N, y)
        y_pred = knn.predict(N_test)
        error = 1 - accuracy_score(y_test, y_pred)

        return error

    @classmethod
    def ft_C1(cls,
              cls_amount: np.ndarray
              ) -> np.ndarray:

        nc = cls_amount.shape[0]
        pc_i = cls_amount / np.sum(cls_amount)
        c1 = -(1.0/np.log(nc)) * np.sum(pc_i*np.log(pc_i))

        # Shuldn't C1 be 1-C1? to match with C2?
        return c1

    @classmethod
    def ft_C2(cls,
              cls_amount: np.ndarray
              ) -> np.ndarray:

        n = np.sum(cls_amount)
        nc = cls_amount.shape[0]
        nc_i = cls_amount
        IR = ((nc-1) / nc) * np.sum(nc_i / (n - (nc_i)))
        c2 = 1 - (1/IR)

        return c2

    @classmethod
    def ft_T2(cls,
              N: np.ndarray
              ) -> np.ndarray:

        m = N.shape[1]
        n = N.shape[0]

        return m/n

    @classmethod
    def ft_T3(cls,
              N: np.ndarray
              ) -> np.ndarray:

        pca = PCA(n_components=0.95)
        pca.fit(N)

        m_ = pca.explained_variance_ratio_.shape[0]
        n = N.shape[0]

        return m_/n
