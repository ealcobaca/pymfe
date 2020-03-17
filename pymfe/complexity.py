"""Module dedicated to extraction of complexity metafeatures."""

import typing as t
import itertools

import numpy as np
import sklearn
import sklearn.pipeline
import scipy.spatial

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
    def precompute_fx(cls, y: t.Optional[np.ndarray] = None,
                      **kwargs) -> t.Dict[str, t.Any]:
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
                  rows corresponds to the distinct classes, and the instances
                  are represented by the columns.
                - ``classes`` (:obj:`np.ndarray`): distinct classes in the
                  fitted target attribute.
                - ``class_freqs`` (:obj:`np.ndarray`): The number of examples
                  in each class. The indices corresponds to the classes.
        """
        precomp_vals = {}  # type: t.Dict[str, t.Any]

        if (y is not None and not {"classes", "class_freqs"}.issubset(kwargs)):
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
        precomp_vals = {}

        if N is not None and "num_attr_pca" not in kwargs:
            pca = sklearn.decomposition.PCA(
                n_components=tx_n_components, random_state=random_state)
            pca.fit(N)

            num_attr_pca = pca.explained_variance_ratio_.shape[0]

            precomp_vals["num_attr_pca"] = num_attr_pca

        return precomp_vals

    @staticmethod
    def _calc_ovo_comb(classes: np.ndarray) -> t.List[t.Tuple]:
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
        ind_less_overlap = np.argmin(feat_overlap_num)

        return ind_less_overlap, feat_overlap_num, feat_overlapped_region

    @classmethod
    def ft_f1(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            cls_inds: t.Optional[np.ndarray] = None,
            class_freqs: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Maximum Fisher's discriminant ratio.

        ...

        This measure is in (0, 1] range.

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
            ...
        """
        classes = None

        if class_freqs is None:
            classes, class_freqs = np.unique(y, return_counts=True)

        if cls_inds is None:
            if classes is None:
                classes = np.unique(y)

            cls_inds = _utils.calc_cls_inds(y, classes)

        mean_global = np.mean(N, axis=0)

        centroids = np.asarray([
            np.mean(N[inds_cur_cls, :], axis=0) for inds_cur_cls in cls_inds
        ], dtype=float)

        _numer = np.sum(
            np.square(centroids - mean_global).T * class_freqs, axis=1)

        _denom = np.sum(np.square([
            N[inds_cur_cls, :] - centroids[cls_ind, :]
            for cls_ind, inds_cur_cls in enumerate(cls_inds)
        ]), axis=(0, 1))

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
            class_freqs: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Directional-vector maximum Fisher's discriminant ratio.

        ...

        This measure is in (0, 1] range.

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
            ...
        """
        if ovo_comb is None or cls_inds is None or class_freqs is None:
            sub_dic = cls.precompute_fx(y=y)
            ovo_comb = sub_dic["ovo_comb"]
            cls_inds = sub_dic["cls_inds"]
            class_freqs = sub_dic["class_freqs"]

        num_attr = N.shape[1]

        df = np.zeros(ovo_comb.shape[0], dtype=float)
        mat_scatter_within = []
        centroids = np.zeros((class_freqs.size, num_attr), dtype=float)

        for cls_ind, inds_cur_cls in enumerate(cls_inds):
            cur_cls_inst = N[inds_cur_cls, :]
            mat_scatter_within.append(
                np.cov(cur_cls_inst, rowvar=False, ddof=1))
            centroids[cls_ind, :] = cur_cls_inst.mean(axis=0)

        for ind, (cls_id_1, cls_id_2) in enumerate(ovo_comb):
            centroid_diff = (
                centroids[cls_id_1, :] - centroids[cls_id_2, :]).reshape(-1, 1)

            total_inst_num = class_freqs[cls_id_1] + class_freqs[cls_id_2]

            W_mat = (
                class_freqs[cls_id_1] * mat_scatter_within[cls_id_1] +
                class_freqs[cls_id_2] * mat_scatter_within[cls_id_2]
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
            cls_inds: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Volume of the overlapping region.

        ...

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
            ...
        """
        if ovo_comb is None or cls_inds is None:
            sub_dic = cls.precompute_fx(y=y)
            ovo_comb = sub_dic["ovo_comb"]
            cls_inds = sub_dic["cls_inds"]

        f4 = np.zeros(ovo_comb.shape[0], dtype=float)

        for ind, (cls_id_1, cls_id_2) in enumerate(ovo_comb):
            N_cls_1 = N[cls_inds[cls_id_1], :]
            N_cls_2 = N[cls_inds[cls_id_2], :]

            maxmax = cls._calc_maxmax(N_cls_1, N_cls_2)
            minmin = cls._calc_minmin(N_cls_1, N_cls_2)
            minmax = cls._calc_minmax(N_cls_1, N_cls_2)
            maxmin = cls._calc_maxmin(N_cls_1, N_cls_2)

            f4[ind] = np.prod(
                np.maximum(0.0, minmax - maxmin) / (maxmax - minmin))

        return f4

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
            sub_dic = cls.precompute_fx(y=y)
            ovo_comb = sub_dic["ovo_comb"]
            cls_inds = sub_dic["cls_inds"]
            class_freqs = sub_dic["class_freqs"]

        f3 = np.zeros(ovo_comb.shape[0], dtype=float)

        for ind, (cls_id_1, cls_id_2) in enumerate(ovo_comb):
            N_cls_1 = N[cls_inds[cls_id_1, :], :]
            N_cls_2 = N[cls_inds[cls_id_2, :], :]

            ind_less_overlap, feat_overlap_num, _ = cls._calc_overlap(
                N=N,
                minmax=cls._calc_minmax(N_cls_1, N_cls_2),
                maxmin=cls._calc_maxmin(N_cls_1, N_cls_2))

            f3[ind] = (feat_overlap_num[ind_less_overlap] /
                       (class_freqs[cls_id_1] + class_freqs[cls_id_2]))

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
            sub_dic = cls.precompute_fx(y=y)
            ovo_comb = sub_dic["ovo_comb"]
            cls_inds = sub_dic["cls_inds"]
            class_freqs = sub_dic["class_freqs"]

        f4 = np.zeros(ovo_comb.shape[0], dtype=float)

        for ind, (cls_id_1, cls_id_2) in enumerate(ovo_comb):
            cls_subset_union = np.logical_or(cls_inds[cls_id_1, :],
                                             cls_inds[cls_id_2, :])

            cls_1 = cls_inds[cls_id_1, cls_subset_union]
            cls_2 = cls_inds[cls_id_2, cls_subset_union]
            N_subset = N[cls_subset_union, :]

            # Search only on remaining features, without copying any data
            valid_attr_inds = np.arange(N_subset.shape[1])
            N_view = N_subset[:, valid_attr_inds]

            while N_view.size > 0:
                N_cls_1, N_cls_2 = N_view[cls_1, :], N_view[cls_2, :]

                # Note: 'feat_overlapped_region' is a boolean vector with
                # True values if the example is in the overlapping region
                ind_less_overlap, _, feat_overlapped_region = (
                    cls._calc_overlap(
                        N=N_view,
                        minmax=cls._calc_minmax(N_cls_1, N_cls_2),
                        maxmin=cls._calc_maxmin(N_cls_1, N_cls_2)))

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

            f4[ind] = subset_size / (class_freqs[cls_id_1] +
                                     class_freqs[cls_id_2])

        # The measure is computed in the literature using the mean. However, it
        # is formulated here as a meta-feature. Therefore, the post-processing
        # should be used to get the mean and other measures as well.
        return f4

    @classmethod
    def ft_l1(cls, N: np.ndarray, y: np.ndarray) -> np.ndarray:
        """TODO.

        ...

        This measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        Returns
        -------
        :obj:`np.ndarray`
            ...
        """

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
            The rows corresponds to each distinct class, and the columns
            corresponds to the instances.

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
           (Cited on page 9). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        """
        if ovo_comb is None or cls_inds is None:
            sub_dic = cls.precompute_fx(y=y)
            ovo_comb = sub_dic["ovo_comb"]
            cls_inds = sub_dic["cls_inds"]

        l2 = np.zeros(ovo_comb.shape[0], dtype=float)

        zscore = sklearn.preprocessing.StandardScaler()

        svc = sklearn.svm.SVC(
            kernel="linear", C=1.0, tol=10e-3, max_iter=int(max_iter))

        pip = sklearn.pipeline.Pipeline([("zscore", zscore), ("svc", svc)])

        for ind, (cls_1, cls_2) in enumerate(ovo_comb):
            cls_intersec = np.logical_or(cls_inds[cls_1, :],
                                         cls_inds[cls_2, :])

            N_subset = N[cls_intersec, :]
            y_subset = cls_inds[cls_1, cls_intersec]

            pip.fit(N_subset, y_subset)
            y_pred = pip.predict(N_subset)

            error = sklearn.metrics.zero_one_loss(
                y_true=y_subset, y_pred=y_pred, normalize=True)

            l2[ind] = error

        # The measure is computed in the literature using the mean. However, it
        # is formulated here as a meta-feature. Therefore, the post-processing
        # should be used to get the mean and other measures as well.
        return l2

    @classmethod
    def ft_l3(cls, N: np.ndarray, y: np.ndarray) -> np.ndarray:
        """TODO.

        ...

        This measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        Returns
        -------
        :obj:`np.ndarray`
            ...
        """

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
           (Cited on page 9-10). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        """
        # 0-1 feature scaling
        N = sklearn.preprocessing.MinMaxScaler(
            feature_range=(0, 1)).fit_transform(N)

        # Compute the distance matrix and the minimum spanning tree
        # using Kruskal's Minimum Spanning Tree algorithm.
        dist_mat = scipy.spatial.distance.cdist(N, N, metric=metric)
        # Note: in the paper, the authors used Prim's algorithm.
        # Our implementation may change it in a future version due to
        # time complexity advantages of Prim's algorithm in this context.

        mst = scipy.sparse.csgraph.minimum_spanning_tree(
            csgraph=np.triu(dist_mat, k=1), overwrite=True)

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
    def ft_n2(cls, N: np.ndarray, y: np.ndarray) -> np.ndarray:
        """TODO.

        ...

        This measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        Returns
        -------
        :obj:`np.ndarray`
            ...
        """

    @classmethod
    def ft_n3(cls, N: np.ndarray, y: np.ndarray) -> np.ndarray:
        """TODO.

        ...

        This measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        Returns
        -------
        :obj:`np.ndarray`
            ...
        """

    @classmethod
    def ft_n4(cls,
              N: np.ndarray,
              y: np.ndarray,
              cls_inds: t.Optional[np.ndarray] = None,
              metric_n4: str = "minkowski",
              p_n4: int = 2,
              n_neighbors_n4: int = 1,
              random_state: t.Optional[int] = None) -> np.ndarray:
        """Compute the non-linearity of the NN Classifier.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows corresponds to each distinct class, and the columns
            corresponds to the instances.

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
        :obj:`np.ndarray`
            Misclassifications of the NN classifier in the interpolated
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
            classes = np.unique(y)
            cls_inds = _utils.calc_cls_inds(y, classes)

        # 0-1 feature scaling
        N = sklearn.preprocessing.MinMaxScaler(
            feature_range=(0, 1)).fit_transform(N)

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

        knn = sklearn.neighbors.KNeighborsClassifier(
            n_neighbors=n_neighbors_n4, p=p_n4, metric=metric_n4).fit(N, y)

        y_pred = knn.predict(N_test)

        misclassifications = np.not_equal(y_test, y_pred).astype(int)

        # The measure is computed in the literature using the mean. However, it
        # is formulated here as a meta-feature. Therefore, the post-processing
        # should be used to get the mean and other measures as well.
        return misclassifications

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
    def ft_t1(cls, N: np.ndarray, y: np.ndarray) -> np.ndarray:
        """TODO.

        ...

        This measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        Returns
        -------
        :obj:`np.ndarray`
            ...
        """

    @classmethod
    def ft_t2(cls, N: np.ndarray) -> float:
        """Compute the average number of features per dimension.

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
           (Cited on page 15). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        """
        if num_attr_pca is None:
            sub_dic = cls.precompute_pca_tx(N=N, random_state=random_state)
            num_attr_pca = sub_dic["num_attr_pca"]

        num_inst = N.shape[0]

        return num_attr_pca / num_inst

    @classmethod
    def ft_t4(cls,
              N: np.ndarray,
              num_attr_pca: t.Optional[int] = None,
              random_state: t.Optional[int] = None) -> float:
        """Compute the ratio of the PCA dimension to the original dimension.

        The components kept in the PCA dimension explains at least 95% of
        the data variance.

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
           (Cited on page 15). Published in ACM Computing Surveys (CSUR),
           Volume 52 Issue 5, October 2019, Article No. 107.
        """
        if num_attr_pca is None:
            sub_dic = cls.precompute_pca_tx(N=N, random_state=random_state)
            num_attr_pca = sub_dic["num_attr_pca"]

        num_attr = N.shape[1]

        return num_attr_pca / num_attr

    @classmethod
    def ft_lsc(cls, N: np.ndarray, y: np.ndarray) -> np.ndarray:
        """TODO.

        ...

        This measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        Returns
        -------
        :obj:`np.ndarray`
            ...
        """

    @classmethod
    def ft_density(cls, N: np.ndarray, y: np.ndarray) -> np.ndarray:
        """TODO.

        ...

        This measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        Returns
        -------
        :obj:`np.ndarray`
            ...
        """

    @classmethod
    def ft_cis_coef(cls, N: np.ndarray, y: np.ndarray) -> np.ndarray:
        """TODO.

        ...

        This measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        Returns
        -------
        :obj:`np.ndarray`
            ...
        """

    @classmethod
    def ft_hubs(cls, N: np.ndarray, y: np.ndarray) -> np.ndarray:
        """TODO.

        ...

        This measure is in [0, 1] range.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        Returns
        -------
        :obj:`np.ndarray`
            ...
        """
