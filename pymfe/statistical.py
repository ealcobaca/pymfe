"""A module dedicated to the extraction of Statistical Metafeatures.

Notes:
    For more information about the metafeatures implemented here,
    check out `Rivolli et al.`_.

References:
    .. _Rivolli et al.:
        "Towards Reproducible Empirical Research in Meta-Learning,"
        Rivolli et al. URL: https://arxiv.org/abs/1808.10406
"""
import typing as t

import numpy as np
import scipy

import pymfe._summary as _summary


class MFEStatistical:
    """Keep methods for metafeatures of ``Statistical`` group.

    The convention adopted for metafeature-extraction related methods
    is to always start with ``ft_`` prefix in order to allow automatic
    method detection. This prefix is predefined within ``_internal``
    module.

    All method signature follows the conventions and restrictions listed
    below:
        1. For independent attribute data, ``X`` means ``every type of attribu-
            te``, ``N`` means ``Numeric attributes only`` and ``C`` stands for
            ``Categorical attributes only``. It is important to note that the
            categorical attribute sets between ``X`` and ``C`` and the numeri-
            cal attribute sets between ``X`` and ``N`` may differ due to data
            transformations, performed while fitting data into MFE model, en-
            abled by, respectively, ``transform_num`` and ``transform_cat``
            arguments from ``fit`` (MFE method).

        2. Only arguments in MFE ``_custom_args_ft`` attribute (set up inside
            ``fit`` method) are allowed to be required method arguments. All
            other arguments must be strictly optional (i.e., has a predefined
            default value).

        3. It is assumed that the user can change any optional argument,
            without any previous verification for both type or value, via
            **kwargs argument of ``extract`` method of MFE class.

        4. The return value of all feature-extraction methods should be
            a single value or a generic Sequence (preferably an np.ndarray)
            type with numeric values.

    There is another type of method adopted for automatic detection. It is ad-
    opted the prefix ``precompute_`` for automatic detection of these methods.
    These methods run while fitting some data into an MFE model automatically,
    and their objective is to precompute some common value shared between more
    than one feature extraction method. This strategy is a trade-off between
    more system memory consumption and speeds up of feature extraction. Their
    return value must always be a dictionary whose keys are possible extra ar-
    guments for both feature extraction methods and other precomputation me-
    thods. Note that there is a share of precomputed values between all valid
    feature-extraction modules (e.g., ``class_freqs`` computed in module ``sta-
    tistical`` can freely be used for any precomputation or feature extraction
    method of module ``landmarking``).
    """

    @classmethod
    def precompute_statistical_class(cls,
                                     y: t.Optional[np.ndarray] = None,
                                     **kwargs) -> t.Dict[str, t.Any]:
        """Precompute distinct classes and its abs. frequencies from ``y``.

        Args:
            y (:obj:`np.ndarray`, optional): the target attribute from fitted
                data.

            **kwargs: additional arguments. May have previously precomputed be-
                fore this method from other precomputed methods, so they can
                help speed up this precomputation.

        Return:
            dict: with following precomputed items:

            - ``classes`` (:obj:`np.ndarray`): distinct classes of ``y``,
                if ``y`` is not :obj:`NoneType`.

            - ``class_freqs`` (:obj:`np.ndarray`): absolute class frequencies
                of ``y``, if ``y`` is not :obj:`NoneType`.
        """
        precomp_vals = {}

        if y is not None and not {"classes", "class_freqs"}.issubset(kwargs):
            classes, class_freqs = np.unique(y, return_counts=True)

            precomp_vals["classes"] = classes
            precomp_vals["class_freqs"] = class_freqs

        return precomp_vals

    @classmethod
    def precompute_statistical_eigen(cls,
                                     N: t.Optional[np.ndarray] = None,
                                     y: t.Optional[np.ndarray] = None,
                                     ddof: int = 1,
                                     **kwargs) -> t.Dict[str, t.Any]:
        """Precompute eigenvalues and eigenvectors of LDA Matrix.

        Args:
            N (:obj:`np.ndarray`, optional): numerical attributes from fitted
                data.

            y (:obj:`np.ndarray`, optional): target attribute from fitted data.

            ddof (:obj:`int`, optional): degrees of freedom of covariance ma-
                trix calculated during LDA.

            **kwargs: additional arguments. May have previously precomputed be-
                fore this method from other precomputed methods, so they can
                help speed up this precomputation.

        Return:
            dict: with following precomputed items:

            - ``eig_vals`` (:obj:`np.ndarray`): array with filtered eigen-
                values of Fisher's Linear Discriminant Analysis Matrix.

            - ``eig_vecs`` (:obj:`np.ndarray`): array with filtered eigen-
                vectors of Fisher's Linear Discriminant Analysis Matrix.

            The following items are used by this method, so they must be
            precomputed too (and, therefore, are also in this return dict):

            - ``classes`` (:obj:`np.ndarray`): distinct classes of ``y``,
                ``y``, if both ``N`` and ``y`` are not :obj:`NoneType`.

            - ``class_freqs`` (:obj:`np.ndarray`): class frequencies of
                ``y``, if both ``N`` and ``y`` are not :obj:`NoneType`.
        """
        precomp_vals = {}

        if (y is not None and N is not None and N.size
                and not {"eig_vals", "eig_vecs"}.issubset(kwargs)):
            classes = kwargs.get("classes")
            class_freqs = kwargs.get("class_freqs")

            if classes is None or class_freqs is None:
                classes, class_freqs = np.unique(y, return_counts=True)

            eig_vals, eig_vecs = MFEStatistical._linear_disc_mat_eig(
                N, y, classes=classes, class_freqs=class_freqs, ddof=ddof)

            _, num_attr = N.shape

            eig_vals, eig_vecs = MFEStatistical._filter_eig_vals(
                num_attr=num_attr,
                num_classes=classes.size,
                eig_vals=eig_vals,
                eig_vecs=eig_vecs)

            precomp_vals["eig_vals"] = eig_vals
            precomp_vals["eig_vecs"] = eig_vecs
            precomp_vals["classes"] = classes
            precomp_vals["class_freqs"] = class_freqs

        return precomp_vals

    @classmethod
    def precompute_statistical_cor_cov(cls,
                                       N: t.Optional[np.ndarray] = None,
                                       ddof: int = 1,
                                       **kwargs) -> t.Dict[str, t.Any]:
        """Precomputes the correlation and covariance matrix of numerical data.

        Be cautious in allowing this precomputation method on huge datasets, as
        this precomputation method may be very memory hungry.

        Args:
            N (:obj:`np.ndarray`, optional): numerical attributes from fitted
                data.

            ddof (:obj:`int`, optional): degrees of freedom of covariance ma-
                trix.

            **kwargs: additional arguments. May have previously precomputed be-
                fore this method from other precomputed methods, so they can
                help speed up this precomputation.

        Return:
            dict: with following precomputed items:

            - ``cov_mat`` (:obj:`np.ndarray`): covariance matrix of ``N``,
                if ``N`` is not :obj:`NoneType`.

            - ``abs_corr_mat`` (:obj:`np.ndarray`): absolute correlation
                matrix of ``N``, if ``N`` is not :obj:`NoneType`.
        """
        precomp_vals = {}

        if N is not None and N.size:
            N = N.astype(float)

            if "cov_mat" not in kwargs:
                precomp_vals["cov_mat"] = np.cov(N, rowvar=False, ddof=ddof)

            if "abs_corr_mat" not in kwargs:
                abs_corr_mat = abs(np.corrcoef(N, rowvar=False))

                if (not isinstance(abs_corr_mat, np.ndarray)
                        and np.isnan(abs_corr_mat)):
                    abs_corr_mat = np.array([np.nan])

                precomp_vals["abs_corr_mat"] = abs_corr_mat

        return precomp_vals

    @classmethod
    def _linear_disc_mat_eig(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            ddof: int = 1,
            classes: t.Optional[np.ndarray] = None,
            class_freqs: t.Optional[np.ndarray] = None,
    ) -> t.Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues/vecs of the Linear Discriminant Analysis Matrix.

        More specifically, the eigenvalues and eigenvectors are calculated from
        matrix S = (Scatter_Within_Mat)^(-1) * (Scatter_Between_Mat).

        Check ``ft_can_cor`` documentation for more in-depth information about
        this matrix.

        Args:
            ddof (:obj:`int`, optional): degrees of freedom of covariance ma-
                trix calculated during LDA.

            classes (:obj:`np.ndarray`, optional): distinct classes of ``y``.

            class_freqs (:obj:`np.ndarray`, optional): absolute class frequen-
                cies of ``y``.

        Return:
            tuple(np.ndarray, np.ndarray): eigenvalues and eigenvectors (in
                this order) of Linear Discriminant Analysis Matrix.
        """

        def compute_scatter_within(
                N: np.ndarray,
                y: np.ndarray,
                class_val_freq: t.Tuple[np.ndarray, np.ndarray],
                ddof: int = 1) -> np.ndarray:
            """Compute Scatter Within matrix. Check doc above for more info."""
            scatter_within = np.array(
                [(cl_frq - 1.0) * np.cov(
                    N[y == cl_val, :], rowvar=False, ddof=ddof)
                 for cl_val, cl_frq in zip(*class_val_freq)]).sum(axis=0)

            return scatter_within

        def compute_scatter_between(
                N: np.ndarray, y: np.ndarray,
                class_val_freq: t.Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
            """Compute Scatter Between matrix. The doc above has more info."""
            class_vals, class_freqs = class_val_freq

            class_means = np.array(
                [N[y == cl_val, :].mean(axis=0) for cl_val in class_vals])

            relative_centers = class_means - N.mean(axis=0)

            scatter_between = np.array([
                cl_frq * np.outer(rc, rc)
                for cl_frq, rc in zip(class_freqs, relative_centers)
            ]).sum(axis=0)

            return scatter_between

        if classes is None or class_freqs is None:
            class_val_freq = np.unique(y, return_counts=True)

        else:
            class_val_freq = (classes, class_freqs)

        N = N.astype(float)

        scatter_within = compute_scatter_within(
            N, y, class_val_freq, ddof=ddof)
        scatter_between = compute_scatter_between(N, y, class_val_freq)

        try:
            scatter_within_inv = np.linalg.inv(scatter_within)

            return np.linalg.eig(
                np.matmul(scatter_within_inv, scatter_between))

        except (np.linalg.LinAlgError, ValueError):
            return np.array([np.nan]), np.array([np.nan])

    @classmethod
    def _filter_eig_vals(
            cls,
            eig_vals: np.ndarray,
            num_attr: int,
            num_classes: int,
            eig_vecs: t.Optional[np.ndarray] = None,
            filter_imaginary: bool = True,
            filter_less_relevant: float = True,
            epsilon: float = 1.0e-8,
    ) -> t.Union[t.Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Get most expressive eigenvalues (higher absolute value).

        This function returns N eigenvalues, such that:

            N <= min(num_class, num_attr)

        Args:
            eig_vals (:obj:`np.ndarray`): eigenvalues to be filtered.

            num_attr (:obj:`int`): number of attributes (columns) in data.

            num_classes (:obj:`int`): number of distinct classes in fitted
                data.

            eig_vecs (:obj:`np.ndarray`, optional): eigenvectors to filter
                alongside eigenvalues.

            filter_imaginary (:obj:`bool`, optional): if True, remove ima-
                ginary valued eigenvalues and its correspondent eigenvec-
                tors.

            filter_less_relevant (:obj:`bool`, optional): if True, remove
                eigenvalues smaller than ``epsilon``.

            epsilon (:obj:`float`, optional): a tiny value used to determi-
                ne ``less relevant`` eigenvalues.
        """
        max_valid_eig = min(num_attr, num_classes)

        if eig_vals.size <= max_valid_eig:
            if eig_vecs is not None:
                return eig_vals, eig_vecs

            return eig_vals

        if eig_vecs is None:
            eig_vals = np.array(
                sorted(eig_vals, key=abs, reverse=True)[:max_valid_eig])
        else:
            eig_vals, eig_vecs = zip(
                *sorted(
                    zip(eig_vals, eig_vecs),
                    key=lambda item: abs(item[0]),
                    reverse=True)[:max_valid_eig])

            eig_vals = np.array(eig_vals)
            eig_vecs = np.array(eig_vecs)

        if not filter_imaginary and not filter_less_relevant:
            if eig_vecs is not None:
                return eig_vals, eig_vecs

            return eig_vals

        indexes_to_keep = np.array(eig_vals.size * [True])

        if filter_imaginary:
            indexes_to_keep = np.logical_and(
                np.isreal(eig_vals), indexes_to_keep)

        if filter_less_relevant:
            indexes_to_keep = np.logical_and(
                abs(eig_vals) > epsilon, indexes_to_keep)

        eig_vals = eig_vals[indexes_to_keep]

        if filter_imaginary:
            eig_vals = eig_vals.real

        if eig_vecs is not None:
            eig_vecs = eig_vecs[indexes_to_keep, :]

            return eig_vals, eig_vecs

        return eig_vals

    @classmethod
    def ft_can_cor(cls,
                   N: np.ndarray,
                   y: np.ndarray,
                   epsilon: float = 1.0e-10,
                   ddof: int = 1,
                   eig_vals: t.Optional[np.ndarray] = None,
                   classes: t.Optional[np.ndarray] = None,
                   class_freqs: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Compute canonical correlations of data.

        The canonical correlations p are defined as shown below:

            p_i = sqrt(lda_eig_i / (1.0 + lda_eig_i))

        Where ``lda_eig_i`` is the ith eigenvalue obtained when solving the ge-
        neralized eigenvalue problem of Linear Discriminant Analysis Scatter
        Matrix S defined as:

            S = (Scatter_Within_Mat)^(-1) * (Scatter_Between_Mat),

        where
            Scatter_Within_Mat = sum((N_c - 1.0) * Covariance(X_c)), ``N_c``
            is the number of instances of class c and X_c are the instances of
            class ``c``. Effectively, this is exactly just the summation of
            all Covariance matrices between instances of the same class with-
            out dividing then by the number of instances.

            Scatter_Between_Mat = sum(N_c * (U_c - U) * (U_c - U)^T), `'N_c``
            is the number of instances of class c, U_c is the mean coordinates
            of instances of class ``c``, and ``U`` is the mean value of coordi-
            nates of all instances in the dataset.

        Args:
            ddof (:obj:`int`, optional): degrees of freedom of covariance ma-
                trix calculated during LDA.

            epsilon (:obj:`float`, optional): a tiny value to prevent di-
                vision by zero.

            eig_vals (:obj:`np.ndarray`, optional): eigenvalues of LDA Matrix
                ``S``, defined above.

            classes (:obj:`np.ndarray`, optional): distinct classes of ``y``.
        """
        if eig_vals is None:
            if classes is None or class_freqs is None:
                classes, class_freqs = np.unique(y, return_counts=True)

            eig_vals, _ = MFEStatistical._linear_disc_mat_eig(
                N, y, classes=classes, class_freqs=class_freqs, ddof=ddof)

            _, num_attr = N.shape

            eig_vals = MFEStatistical._filter_eig_vals(
                eig_vals=eig_vals, num_attr=num_attr, num_classes=classes.size)

        if not isinstance(eig_vals, np.ndarray):
            eig_vals = np.array(eig_vals)

        return (eig_vals / (epsilon + 1.0 + eig_vals))**0.5

    @classmethod
    def ft_gravity(cls,
                   N: np.ndarray,
                   y: np.ndarray,
                   norm_ord: t.Union[int, float] = 2,
                   classes: t.Optional[np.ndarray] = None,
                   class_freqs: t.Optional[np.ndarray] = None) -> float:
        """Computes the distance between minority and majority classes center of mass.

        The center of mass of a class is the average value of each attribute
        between instances of the same class.

        The majority and minority classes cannot be the same, even if every
        class has the same number of instances.

        Args:
            norm_ord (:obj:`numeric`): Minkowski Distance parameter. Minkowski
                Distance has the following popular cases for this argument va-
                lue:

                +-----------+---------------------------+
                |norm_ord   | Distance name             |
                +-----------+---------------------------+
                |-> -inf    | Min value                 |
                |1.0        | Manhattan/City Block      |
                |2.0        | Euclidean                 |
                |-> +inf    | Max value (infinite norm) |
                +-----------+---------------------------+

        Raises:
            ValueError: if ``norm_ord`` is not numeric.
        """
        if classes is None or class_freqs is None:
            classes, class_freqs = np.unique(y, return_counts=True)

        class_freq_most, _ = max(zip(classes, class_freqs), key=lambda x: x[1])

        class_freq_most_ind = np.where(class_freq_most == classes)[0]

        classes = np.delete(classes, class_freq_most_ind)
        class_freqs = np.delete(class_freqs, class_freq_most_ind)

        class_freq_least, _ = min(
            zip(classes, class_freqs), key=lambda x: x[1])

        center_freq_class_most = N[y == class_freq_most, :].mean(axis=0)
        center_freq_class_least = N[y == class_freq_least, :].mean(axis=0)

        return np.linalg.norm(
            center_freq_class_most - center_freq_class_least, ord=norm_ord)

    @classmethod
    def ft_cor(cls, N: np.ndarray,
               abs_corr_mat: t.Optional[np.ndarray] = None) -> np.ndarray:
        """The absolute value of the correlation of distinct column pairs."""
        if abs_corr_mat is None:
            abs_corr_mat = abs(np.corrcoef(N, rowvar=False))

        if not isinstance(abs_corr_mat, np.ndarray) and np.isnan(abs_corr_mat):
            return np.array([np.nan])

        res_num_rows, _ = abs_corr_mat.shape

        inf_triang_vals = abs_corr_mat[np.tril_indices(res_num_rows, k=-1)]

        return abs(inf_triang_vals)

    @classmethod
    def ft_cov(cls,
               N: np.ndarray,
               ddof: int = 1,
               cov_mat: t.Optional[np.ndarray] = None) -> np.ndarray:
        """The absolute value of the covariance of distinct column pairs.

        Args:
            ddof (:obj:`int`, optional): degrees of freedom for covariance ma-
                trix.

            cov_mat (:obj:`np.ndarray`, optional): covariance matrix of ``N``.
                Argument meant to exploit precomputations. Note that this ar-
                gument value is not the same as this method return value, as
                it only returns the lower-triangle values from ``cov_mat``.
        """
        if cov_mat is None:
            cov_mat = np.cov(N, rowvar=False, ddof=ddof)

        res_num_rows, _ = cov_mat.shape

        inf_triang_vals = cov_mat[np.tril_indices(res_num_rows, k=-1)]

        return abs(inf_triang_vals)

    @classmethod
    def ft_nr_disc(cls,
                   N: np.ndarray,
                   y: np.ndarray,
                   epsilon: float = 1.0e-10,
                   eig_vals: t.Optional[np.ndarray] = None,
                   classes: t.Optional[np.ndarray] = None,
                   class_freqs: t.Optional[np.ndarray] = None) -> float:
        """Compute the number of canonical corr. between each attr. and class.

        This method return value is effectively the size of the return value
        of ``ft_can_cor`` method. Check its documentation for more in-depth
        details.
        """
        can_cor = MFEStatistical.ft_can_cor(
            N=N,
            y=y,
            epsilon=epsilon,
            eig_vals=eig_vals,
            classes=classes,
            class_freqs=class_freqs)

        if isinstance(can_cor, np.ndarray):
            return can_cor.size

        return np.nan

    @classmethod
    def ft_eigenvalues(cls,
                       N: np.ndarray,
                       ddof: int = 1,
                       cov_mat: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Returns the eigenvalues of ``N`` covariance matrix.

        Args:
            ddof (:obj:`int`, optional): degrees of freedom for covariance ma-
                trix.

            cov_mat (:obj:`np.ndarray`, optional): covariance matrix of ``N``.
                Argument meant to exploit precomputations.
        """
        if cov_mat is None:
            cov_mat = np.cov(N, rowvar=False, ddof=ddof)

        try:
            eigvals = np.linalg.eigvals(cov_mat)

        except (np.linalg.LinAlgError, ValueError):
            return np.array([np.nan])

        return eigvals

    @classmethod
    def ft_g_mean(cls,
                  N: np.ndarray,
                  allow_zeros: bool = False,
                  epsilon: float = 1.0e-10) -> np.ndarray:
        """Computes the geometric mean of each attribute in ``N``.

        Args:
            allow_zeros (:obj:`bool`): if True, then the geometric mean of all
                attributes with zero values is set to zero. Otherwise, is set
                to :obj:`np.nan` these values.

            epsilon (:obj:`float`): a small value which all values with absolu-
                te value lesser than it is considered zero-valued.
        """
        if N.size == 0:
            return np.array([np.nan])

        min_values = N.min(axis=0)

        if allow_zeros:
            cols_invalid = min_values < 0.0
            cols_zero = 0.0 <= abs(min_values) < epsilon
            cols_valid = np.logical_not(np.logical_or(cols_invalid, cols_zero))

        else:
            cols_invalid = min_values <= epsilon
            cols_valid = np.logical_not(cols_invalid)

        _, num_col = N.shape
        g_mean = np.zeros(num_col)

        g_mean[cols_valid] = scipy.stats.gmean(N[:, cols_valid], axis=0)

        g_mean[cols_invalid] = np.nan

        return g_mean

    @classmethod
    def ft_h_mean(cls, N: np.ndarray, epsilon: float = 1.0e-8) -> np.ndarray:
        """The harmonic mean of each attribute in ``N``.

        Args:
            epsilon (:obj:`float`, optional): a tiny value to prevent di-
                vision by zero.
        """
        try:
            return scipy.stats.hmean(N + epsilon, axis=0)

        except ValueError:
            return np.array([np.nan])

    @classmethod
    def ft_iq_range(cls, N: np.ndarray) -> np.ndarray:
        """Compute the interquartile range (IQR) of each attribute in ``N``."""
        return scipy.stats.iqr(N, axis=0)

    @classmethod
    def ft_kurtosis(cls, N: np.ndarray, method: int = 3,
                    bias: bool = True) -> np.ndarray:
        """Compute the kurtosis of each attribute in ``N``.

        Args:
            method (:obj:`int`, optional): defines the strategy used for esti-
                mate data kurtosis. Used for total compatibility with R package
                ``e1071``. The options must be one of the following:

                +--------+-----------------------------------------------+
                |Option  | Formula                                       |
                +--------+-----------------------------------------------+
                |1       | Kurt_1 = m_4 / m_2**2 - 3                     |
                |        | (default of ``scipy.stats``)                  |
                +--------+-----------------------------------------------+
                |2       | Kurt_2 = ((n+1) * Kurt_1 + 6) * (n-1) / f_2   |
                |        | f_2 = ((n-2)*(n-3))                           |
                +--------+-----------------------------------------------+
                |3       | Kurt_3 = m_4 / s**4 - 3                       |
                |        |        = (Kurt_1+3) * (1 - 1/n)**2 - 3        |
                +--------+-----------------------------------------------+

                Where ``n`` is the number of elements in ``values``, ``s`` is
                the standard deviation of ``values`` and ``m_i`` is the ith
                statistical momentum of ``values``.

                Note that if the selected method is unable to be calculated due
                to division by zero, then the first method is used instead.

            bias (:obj:`bool`): If False, then the calculations are corrected
                for statistical bias.
        """
        kurt_arr = np.apply_along_axis(
            func1d=_summary.sum_kurtosis,
            axis=0,
            arr=N,
            method=method,
            bias=bias)

        return kurt_arr

    @classmethod
    def ft_mad(cls, N: np.ndarray, factor: float = 1.4826) -> np.ndarray:
        """Computes Median Absolute Deviation (MAD) adjusted by a ``factor``.

        Args:
            factor (:obj:`float`): multiplication factor for output correction.
                The default ``factor`` is 1.4826 since it is an approximated
                result of MAD of a normally distributed data (with any mean and
                standard deviation of 1.0), so it makes this method result com-
                parable with this sort of data.
        """
        median_dev = abs(N - np.median(N, axis=0))
        return np.median(median_dev, axis=0) * factor

    @classmethod
    def ft_max(cls, N: np.ndarray) -> np.ndarray:
        """Get the maximum value from each ``N`` attribute."""
        return N.max(axis=0)

    @classmethod
    def ft_mean(cls, N: np.ndarray) -> np.ndarray:
        """Returns the mean value of each ``N`` attribute."""
        return N.mean(axis=0)

    @classmethod
    def ft_median(cls, N: np.ndarray) -> np.ndarray:
        """Get the median value from each ``N`` attribute."""
        return np.median(N, axis=0)

    @classmethod
    def ft_min(cls, N: np.ndarray) -> np.ndarray:
        """Get the minimum value from each ``N`` attribute."""
        return N.min(axis=0)

    @classmethod
    def ft_nr_cor_attr(cls,
                       N: np.ndarray,
                       threshold: float = 0.5,
                       normalize: bool = True,
                       epsilon: float = 1.0e-8,
                       abs_corr_mat: t.Optional[np.ndarray] = None
                       ) -> t.Union[int, float]:
        """The number of attribute pairs with corr. eq. to or greater than a threshold.

        Args:
            threshold (:obj:`float`, optional): a value of the threshold, whe-
                re correlation is assumed to be strong if its absolute value is
                equal or greater than it.

            normalize (:obj:`bool`, optional): if True, the result is normali-
                zed by a factor of 2/(d*(d-1)), where ``d`` is number of attri-
                butes (columns) in ``N``.

            epsilon (:obj:`float`, optional): a tiny value to prevent division
                by zero.

            abs_corr_mat (:obj:`np.ndarray`, optional): absolute correlation
                matrix of ``N``. Argument used to exploit precomputations.
        """
        abs_corr_vals = MFEStatistical.ft_cor(N, abs_corr_mat=abs_corr_mat)

        _, num_attr = N.shape

        norm_factor = 1

        if normalize:
            norm_factor = 2.0 / (epsilon + num_attr * (num_attr - 1.0))

        return sum(abs_corr_vals >= threshold) * norm_factor

    @classmethod
    def ft_nr_norm(cls,
                   N: np.ndarray,
                   method: str = "shapiro-wilk",
                   threshold: float = 0.05,
                   failure: str = "soft",
                   max_samples: int = 5000) -> t.Union[float, int]:
        """The number of attributes normally distributed based in some ``method``.

        Args:
            method (:obj:`str`, optional): select the normality test to be exe-
                cuted. This argument must assume one of the options shown be-
                low:

                - ``shapiro-wilk``: directly from `shapiro`_: ``the Shapiro-
                    Wilk test tests the null hypothesis that the data was drawn
                    from a normal distribution.``

                - ``dagostino-pearson``: directly from `normaltest`_: ``It is
                    based on D'Agostino and Pearson's, test that combines skew
                    and kurtosis to produce an omnibus test of normality.``.

                - ``anderson-darling``: directly from `anderson`_: The Ander-
                    son-Darling tests the null hypothesis that a sample is
                    drawn from a population that follows a particular distribu-
                    tion.`` In this method context, that ``particular distribu-
                    tion`` is fixed in the normal/gaussian.

                - ``all``: perform all tests cited above. To consider an attri-
                    bute normaly distributed all test results are taken into
                    account with equal weight. Check ``failure`` argument for
                    more information.

            threshold (:obj:`float`, optional): level of significance used to
                reject the null hypothesis of normality tests.

            failure (:obj:`str`, optional): used only if ``method`` argument
                value is ``all``. This argument must assumed one value between
                ``soft`` or ``hard``. If ``soft``, then if a single test can`t
                have its null hypothesis (of the normal/Gaussian distribution
                of the attribute data) rejected for some attribute, then that
                attribute is considered normally distributed. If ``hard``, then
                is necessary the rejection of the null hypothesis of every sin-
                gle normality test to consider the attribute normally distribu-
                ted.

            max_samples (:obj:`int`, optional): max samples used while perfor-
                ming the normality tests. Shapiro-Wilks test p-value may not
                be accurate when sample size is higher than 5000. Note that
                the instances are NOT shuffled before doing this cutoff. This
                means that the very first ``max_samples`` instances of the da-
                taset ``N`` will be considered in the statistical tests.

        Returns:
            int: the number of normally distributed attributes based on the
                ``method``. If ``max_samples`` is non-positive, :obj:`np.nan`
                is returned instead.

        Raises:
            ValueError: if ``method`` or ``failure`` is not a valid option.

        References:
            .. _shapiro: :obj:`scipy.stats.shapiro` documentation.
            .. _normaltest: :obj:`scipy.stats.normaltest` documentation.
            .. _anderson: :obj:`scipy.stats.anderson` documentation.
        """
        accepted_tests = (
            "shapiro-wilk",
            "dagostino-pearson",
            "anderson-darling",
            "all",
        )

        if method not in accepted_tests:
            raise ValueError("Unknown method {0}. Select one between"
                             "{1}".format(method, accepted_tests))

        if failure not in ("hard", "soft"):
            raise ValueError('"failure" argument must be either "soft"'
                             'or "hard" (got "{}").'.format(failure))

        if max_samples <= 0:
            return np.nan

        num_inst, num_attr = N.shape

        max_row_index = min(max_samples, num_inst)

        test_results = []

        if method in ("shapiro-wilk", "all"):
            _, p_values_shapiro = np.apply_along_axis(
                func1d=scipy.stats.shapiro, axis=0, arr=N[:max_row_index, :])

            test_results.append(p_values_shapiro > threshold)

        if method in ("dagostino-pearson", "all"):
            _, p_values_dagostino = scipy.stats.normaltest(
                N[:max_row_index, :], axis=0)

            test_results.append(p_values_dagostino > threshold)

        if method in ("anderson-darling", "all"):
            anderson_stats = np.repeat(False, num_attr)

            for attr_ind, attr_vals in enumerate(N[:max_row_index, :].T):
                stat_value, crit_values, signif_levels = scipy.stats.anderson(
                    attr_vals, dist="norm")

                # As scipy.stats.anderson gives critical values for fixed
                # significance levels, then the strategy adopted is to use
                # the nearest possible from the given threshold as an esti-
                # mator.
                stat_index = np.argmin(abs(signif_levels - threshold))
                crit_val = crit_values[stat_index]

                anderson_stats[attr_ind] = stat_value <= crit_val

            test_results.append(anderson_stats)

        if failure == "soft":
            attr_is_normal = np.any(test_results, axis=0)

        else:
            attr_is_normal = np.all(test_results, axis=0)

        return sum(attr_is_normal)

    @classmethod
    def ft_nr_outliers(cls, N: np.ndarray, whis: float = 1.5) -> int:
        """Calculate the number of attributes which has at least one outlier value.

        An attribute has outlier if some value is outside the closed interval
        [first_quartile - WHIS * IQR, third_quartile + WHIS * IQR], where IQR
        is the Interquartile Range (third_quartile - first_quartile), and WHIS
        value is typically ``1.5``.

        Args:
            whis (:obj:`float`): a factor to multiply IQR and set up non-outli-
            er interval (as stated above). Higher values make the interval more
            significant, thus increasing the tolerance against outliers, where
            lower values decrease non-outlier interval and, therefore, creates
            less tolerance against possible outliers.
        """
        v_min, q_1, q_3, v_max = np.percentile(N, (0, 25, 75, 100), axis=0)

        whis_iqr = whis * (q_3 - q_1)

        cut_low = q_1 - whis_iqr
        cut_high = q_3 + whis_iqr

        return sum(np.logical_or(cut_low > v_min, cut_high < v_max))

    @classmethod
    def ft_range(cls, N: np.ndarray) -> np.ndarray:
        """Compute the range (max - min) of each attribute in ``N``."""
        return np.ptp(N, axis=0)

    @classmethod
    def ft_sd(cls, N: np.ndarray, ddof: int = 1) -> np.ndarray:
        """Compute the standard deviation of each attribute in ``N``.

        Args:
            ddof (:obj:`float`): degrees of freedom for standard deviation.
        """
        sd_array = N.std(axis=0, ddof=ddof)

        sd_array = np.array(
            [np.nan if np.isinf(val) else val for val in sd_array])

        return sd_array

    @classmethod
    def ft_sd_ratio(cls,
                    N: np.ndarray,
                    y: np.ndarray,
                    epsilon: float = 1.0e-8,
                    ddof: int = 1,
                    classes: t.Optional[np.ndarray] = None,
                    class_freqs: t.Optional[np.ndarray] = None) -> float:
        """Perform a statistical test for homogeneity of covariances.

        Args:
            epsilon (:obj:`float`, optional): a tiny value to prevent division
                by zero.

            ddof (:obj:`int`, optional): degrees of freedom for covariance ma-
                trix, calculated during this test.

            classes (:obj:`np.ndarray`, optional): all distinct classes in tar-
                get attribute ``y``. Used to exploit precomputations.

            class_freqs (:obj:`np.ndarray`, optional): absolute frequencies of
                each distinct class in target attribute ``y`` or ``classes``.
                If ``classes`` is given, then this argument must be paired with
                it by index.

        Notes:
            For details about how this test is applied, check out `Rivolli
            et al.`_ (pag. 32).

        References:
            .. _Rivolli et al.:
                "Towards Reproducible Empirical Research in Meta-Learning,"
                Rivolli et al. URL: https://arxiv.org/abs/1808.10406
        """

        def calc_sample_cov_mat(N, y, epsilon, ddof):
            """Calculate the Sample Covariance Matrix for each class."""
            sample_cov_matrices = np.array([
                np.cov(N[y == cl, :] + epsilon, rowvar=False, ddof=ddof)
                for cl in classes
            ])

            return np.flip(np.flip(sample_cov_matrices, 0), 1)

        def calc_pooled_cov_mat(sample_cov_matrices: np.ndarray,
                                vec_weight: np.ndarray, num_inst: int,
                                num_classes: int) -> np.ndarray:
            """Calculate the Pooled Covariance Matrix."""
            pooled_cov_mat = np.array([
                weight * S_i
                for weight, S_i in zip(vec_weight, sample_cov_matrices)
            ]).sum(axis=0) / (num_inst - num_classes)

            return pooled_cov_mat

        def calc_gamma_factor(num_col, num_classes, num_inst, epsilon):
            """Calculate the gamma factor which adjust the output."""
            gamma = 1.0 - (
                (2.0 * num_col**2.0 + 3.0 * num_col - 1.0) /
                (epsilon + 6.0 * (num_col + 1.0) *
                 (num_classes - 1.0))) * (sum(1.0 / vec_weight) - 1.0 /
                                          (epsilon + num_inst - num_classes))
            return gamma

        def calc_m_factor(sample_cov_matrices: np.ndarray,
                          pooled_cov_mat: np.ndarray, num_inst: int,
                          num_classes: int, gamma: float,
                          vec_weight: np.ndarray) -> float:
            """Calculate the M factor."""
            vec_logdet = [
                np.math.log(epsilon + np.linalg.det(S_i))
                for S_i in sample_cov_matrices
            ]

            m_factor = (gamma * ((num_inst - num_classes) * np.math.log(
                np.linalg.det(pooled_cov_mat)) - np.dot(
                    vec_weight, vec_logdet)))

            return m_factor

        num_inst, num_col = N.shape

        if classes is None or class_freqs is None:
            classes, class_freqs = np.unique(y, return_counts=True)

        num_classes = classes.size

        sample_cov_matrices = calc_sample_cov_mat(N, y, epsilon, ddof)

        vec_weight = class_freqs - 1.0 + epsilon

        pooled_cov_mat = calc_pooled_cov_mat(sample_cov_matrices, vec_weight,
                                             num_inst, num_classes)

        gamma = calc_gamma_factor(num_col, num_classes, num_inst, epsilon)

        try:
            m_factor = calc_m_factor(sample_cov_matrices, pooled_cov_mat,
                                     num_inst, num_classes, gamma, vec_weight)

        except np.linalg.LinAlgError:
            return np.nan

        return np.exp(
            m_factor / (epsilon + num_col * (num_inst - num_classes)))

    @classmethod
    def ft_skewness(cls, N: np.ndarray, method: int = 3,
                    bias: bool = True) -> np.ndarray:
        """Compute the skewness for each attribute in ``N``.

        Args:
            method (:obj:`int`, optional): defines the strategy used for es-
                timate data skewness. This argument is used fo compatibility
                with R package ``e1071``. The options must be one of the fol-
                lowing:

                +--------+-----------------------------------------------+
                |Option  | Formula                                       |
                +--------+-----------------------------------------------+
                |1       | Skew_1 = m_3 / m_2**(3/2)                     |
                |        | (default of ``scipy.stats``)                  |
                +--------+-----------------------------------------------+
                |2       | Skew_2 = Skew_1 * sqrt(n(n-1)) / (n-2)        |
                +--------+-----------------------------------------------+
                |3       | Skew_3 = m_3 / s**3 = Skew_1 ((n-1)/n)**(3/2) |
                +--------+-----------------------------------------------+

                Where ``n`` is the number of elements in dataset, ``m_i`` is
                the ith momentum of the attribute, and ``s`` is the standard
                deviation of the attribute.

                Note that if the selected method is unable to be calculated
                due to division by zero, then the first method will be used
                instead.

            bias (:obj:`bool`, optional): If False, then the calculations are
                corrected for statistical bias.
        """
        skew_arr = np.apply_along_axis(
            func1d=_summary.sum_skewness,
            axis=0,
            arr=N,
            bias=bias,
            method=method)

        return skew_arr

    @classmethod
    def ft_sparsity(cls,
                    X: np.ndarray,
                    normalize: bool = True,
                    epsilon: float = 1.0e-8) -> np.ndarray:
        """Compute (possibly normalized) sparsity metric for each attribute.

        Sparsity ``S`` of a vector ``v`` of numeric values is defined as

            S(v) = (1.0 / (n - 1.0)) * ((n / phi(v)) - 1.0),

        where
            - ``n`` is the number of instances in dataset ``X``.
            - ``phi(v)`` is the number of distinct values in ``v``.

        Args:
            normalize (:obj:`bool`, optional): if True, then the output will be
                S(v) as shown above. Otherwise, the output is not be multiplied
                by the ``(1.0 / (n - 1.0))`` factor (i.e. new output is defined
                as S'(v) = ((n / phi(v)) - 1.0)).

            epsilon (:obj:`float`, optional): a small value to prevent division
                by zero.
        """

        ans = np.array([attr.size / np.unique(attr).size for attr in X.T])

        num_inst, _ = X.shape

        norm_factor = 1.0
        if normalize:
            norm_factor = 1.0 / (epsilon + num_inst - 1.0)

        return (ans - 1.0) * norm_factor

    @classmethod
    def ft_t_mean(cls, N: np.ndarray, pcut: float = 0.2) -> np.ndarray:
        """Compute the trimmed mean of each attribute in ``N``.

        Args:
            pcut (:obj:`float`): percentage of cut from both the ``lower`` and
                ``higher`` values. This value should be in interval [0.0, 0.5),
                where if 0.0 the return value is the default mean calculation.
                If this argument is not in mentioned interval, then the return
                value is :obj:`np.nan` instead.
        """
        if not 0 <= pcut < 0.5:
            return np.array([np.nan])

        return scipy.stats.trim_mean(N, proportiontocut=pcut)

    @classmethod
    def ft_var(cls, N: np.ndarray, ddof: int = 1) -> np.ndarray:
        """Compute the variance of each attribute in ``N``.

        Args:
            ddof (:obj:`float`): degrees of freedom for variance.
        """
        var_array = N.var(axis=0, ddof=ddof)

        var_array = np.array(
            [np.nan if np.isinf(val) else val for val in var_array])

        return var_array

    @classmethod
    def ft_w_lambda(cls,
                    N: np.ndarray,
                    y: np.ndarray,
                    ddof: int = 1,
                    eig_vals: t.Optional[np.ndarray] = None,
                    classes: t.Optional[np.ndarray] = None,
                    class_freqs: t.Optional[np.ndarray] = None) -> float:
        """Compute the Wilks' Lambda value.

        The Wilk's Lambda L is calculated as:

            L = prod(1.0 / (1.0 + lda_eig_i))

        Where ``lda_eig_i`` is the ith eigenvalue obtained when solving the ge-
        neralized eigenvalue problem of Linear Discriminant Analysis Scatter
        Matrix. Check ``ft_can_cor`` documentation for more in-depth informati-
        on about this value.

        Args:
            ddof (:obj:`int`, optional): degrees of freedom of covariance ma-
                trix calculated during LDA.

            eig_vals (:obj:`np.ndarray`, optional): eigenvalues of LDA matrix.
                This argument is used to exploit precomputations.

            classes (:obj:`np.ndarray`, optional): all distinct classes in tar-
                get attribute ``y``. Used to exploit precomputations.

            class_freqs (:obj:`np.ndarray`, optional): absolute frequencies of
                each distinct class in target attribute ``y`` or ``classes``.
                If ``classes`` is given, then this argument must be paired with
                it by index.
        """
        if eig_vals is None:
            if classes is None or class_freqs is None:
                classes, class_freqs = np.unique(y, return_counts=True)

            eig_vals, _ = MFEStatistical._linear_disc_mat_eig(
                N, y, classes=classes, class_freqs=class_freqs, ddof=ddof)

            _, num_attr = N.shape

            eig_vals = MFEStatistical._filter_eig_vals(
                eig_vals=eig_vals, num_attr=num_attr, num_classes=classes.size)

        if not isinstance(eig_vals, np.ndarray):
            eig_vals = np.array(eig_vals)

        if eig_vals.size == 0:
            return np.nan

        return np.prod(1.0 / (1.0 + eig_vals))
