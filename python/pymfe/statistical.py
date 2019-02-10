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

import _summary


class MFEStatistical:
    """Keep methods for metafeatures of ``Statistical`` group.

    The convention adopted for metafeature-extraction related methods
    is to always start with ``ft_`` prefix in order to allow automatic
    method detection. This prefix is predefined within ``_internal``
    module.

    All method signature follows the conventions and restrictions listed
    below:
        1. For independent attribute data, ``X`` means ``every type of
            attribute``, ``N`` means ``Numeric attributes only`` and ``C``
            stands for ``Categorical attributes only``.

        2. Only ``X``, ``y``, ``N``, ``C`` and ``splits`` are allowed
            to be required method arguments. All other arguments must be
            strictly optional (i.e. has a predefined default value).

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
                                     **kwargs) -> t.Dict[str, t.Any]:
        """Precompute eigenvalues and eigenvectors of LDA Matrix.

        Args:
            N (:obj:`np.ndarray`, optional): numerical attributes from fitted
                data.

            y (:obj:`np.ndarray`, optional): target attribute from fitted data.

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

        if (y is not None and N is not None
                and not {"eig_vals", "eig_vecs"}.issubset(kwargs)):
            classes = kwargs.get("classes")
            class_freqs = kwargs.get("class_freqs")

            if classes is None or class_freqs is None:
                classes, class_freqs = np.unique(y, return_counts=True)

            eig_vals, eig_vecs = MFEStatistical._linear_disc_mat_eig(
                N, y, classes=classes, class_freqs=class_freqs)

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
                                       **kwargs) -> t.Dict[str, t.Any]:
        """Precomputes the correlation and covariance matrix of numerical data.

        Be cautious in allowing this precomputation method on huge datasets, as
        this precomputation method may be very memory hungry.

        Args:
            N (:obj:`np.ndarray`, optional): numerical attributes from fitted
                data.

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

        if N is not None:
            N = N.astype(float)

            if "cov_mat" not in kwargs:
                precomp_vals["cov_mat"] = np.cov(N, rowvar=False)

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
            classes: t.Optional[np.ndarray] = None,
            class_freqs: t.Optional[np.ndarray] = None,
    ) -> t.Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues/vecs of the Linear Discriminant Analysis Matrix.

        More specifically, the eigenvalues and eigenvectors are calculated from
        matrix S = (Scatter_Within_Mat)^(-1) * (Scatter_Between_Mat).

        Check ``ft_can_cor`` documentation for more in-depth information about
        this matrix.

        Args:
            classes (:obj:`np.ndarray`, optional): distinct classes of ``y``.

            class_freqs (:obj:`np.ndarray`, optional): absolute class frequen-
                cies of ``y``.

        Return:
            tuple(np.ndarray, np.ndarray): eigenvalues and eigenvectors (in
                this order) of Linear Discriminant Analysis Matrix.
        """

        def compute_scatter_within(
                N: np.ndarray, y: np.ndarray,
                class_val_freq: t.Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
            """Compute Scatter Within matrix. Check doc above for more info."""
            scatter_within = np.array(
                [(cl_frq - 1.0) * np.cov(N[y == cl_val, :], rowvar=False)
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

        scatter_within = compute_scatter_within(N, y, class_val_freq)
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

            epsilon (:obj:`float`, optional): a tiny value used to
                determine ``less relevant`` eigenvalues.
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
                   eig_vals: t.Optional[np.ndarray] = None,
                   classes: t.Optional[np.ndarray] = None,
                   class_freqs: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Compute canonical correlations of data.

        The canonical correlations p are defined as shown below:

            p_i = sqrt(lda_eig_i / (1.0 + lda_eig_i))

        Where ``lda_eig_i`` is the ith eigenvalue of Linear Discriminant Ana-
        lysis Matrix S defined as:

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
                N, y, classes=classes, class_freqs=class_freqs)

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
    def ft_cov(cls, N: np.ndarray,
               cov_mat: t.Optional[np.ndarray] = None) -> np.ndarray:
        """The absolute value of the covariance of distinct column pairs.

        Args:
            cov_mat (:obj:`np.ndarray`, optional): covariance matrix of ``N``.
                Argument meant to exploit precomputations. Note that this ar-
                gument value is not the same as this method return value, as
                it only returns the lower-triangle values from ``cov_mat``.
        """
        if cov_mat is None:
            cov_mat = np.cov(N, rowvar=False)

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
                       cov_mat: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Returns the eigenvalues of ``N`` covariance matrix.

        Args:
            cov_mat (:obj:`np.ndarray`, optional): covariance matrix of ``N``.
                Argument meant to exploit precomputations.
        """
        if cov_mat is None:
            cov_mat = np.cov(N, rowvar=False)

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

        g_mean[cols_valid] = scipy.stats.mstats.gmean(N[:, cols_valid], axis=0)

        g_mean[cols_invalid] = np.nan

        return g_mean

    @classmethod
    def ft_h_mean(cls, N: np.ndarray, epsilon: float = 1.0e-8) -> np.ndarray:
        """The harmonic mean of each attribute in ``N``.

        Args:
            epsilon (:obj:`float`, optional): a tiny value to prevent di-
                vision by zero.
        """
        return scipy.stats.mstats.hmean(N + epsilon, axis=0)

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
    def ft_nr_norm(cls, N: np.ndarray,
                   method: str = "shapiro-wilk",
                   threshold: float = 0.05,
                   max_samples: int = 5000) -> int:
        """The number of attributes normally distributed based in ``method``.

        Args:
            method (:obj:`str`, optional): select the normality test to be exe-
                cuted. This argument must assume one of the options shown be-
                low:
                - ``shapiro-wilk``: directly from the ``scipy.stats.shapiro``
                    documentation: ``the Shapiro-Wilk test tests the null hy-
                    pothesis that the data was drawn from a normal distribu-
                    tion.``
                - ``dagostino-pearson``: directly from the ``scipy.stats.nor-
                    maltest`` documentation: ``It is based on D'Agostino and
                    Pearson's, test that combines skew and kurtosis to produce
                    an omnibus test of normality.``
                - ``anderson-darling``: ...
                - ``all``: perform all tests cited above. An attribute is con-
                    sidered normally distributed when rejecting a null hypothe-
                    sis of any test.

            threshold (:obj:`float`, optional): level of significance used to
                reject the null hypothesis.

            max_samples (:obj:`int`, optional): max samples used while perfor-
                ming the normality tests. Shapiro-Wilks test p-value may not
                be accurate when sample size is higher than 5000.

        Returns:
            int: the number of normally distributed attributes based on
                ``method``.

        Raises:
            ValueError: if ``method`` is not a valid option.
        """
        ACCEPTED_TESTS = (
            "shapiro-wilk",
            "dagostino-pearson",
            "anderson-darling",
            "all",
        )

        if method not in ACCEPTED_TESTS:
            raise ValueError("Unknown method {0}. Select one between"
                             "{1}".format(method, ACCEPTED_TESTS))

        num_inst, num_attr = N.shape
        max_row_index = min(max_samples, num_inst)

        attr_is_normal = np.repeat(False, num_attr)

        if method in ("shapiro-wilk", "all"):
            _, p_values_shapiro = np.apply_along_axis(
                func1d=scipy.stats.shapiro,
                axis=0,
                arr=N[:max_row_index, :])

            attr_is_normal[p_values_shapiro > threshold] = True

        if method in ("dagostino-pearson", "all"):
            _, p_values_dagostino = scipy.stats.normaltest(
                N[:max_row_index, :],
                axis=0)

            attr_is_normal[p_values_dagostino > threshold] = True

        """
        if method in ("anderson-darling", "all"):
            _, p_values_anderson, _ = np.apply_along_axis(
                func1d=scipy.stats.anderson,
                axis=0,
                arr=N[:max_row_index, :],
                dist="norm")

            attr_is_normal[p_values_anderson > threshold] = True
        """
        return sum(attr_is_normal)

    @classmethod
    def ft_nr_outliers(cls, N: np.ndarray, whis: float = 1.5) -> int:
        """Calculate number of attribute which has at least one outlier value.

        An attribute has outlier if some value is outside the closed in-
        terval [first_quartile - WHIS * IQR, third_quartile + WHIS * IQR],
        where IQR is the Interquartile Range (third_quartile - first_quartile),
        and WHIS is tipically `1.5`.

        Args:
            whis (:obj:`float`): factor to multiply IQR and set up non-outlier
                interval (as stated above). Higher values make the interval
                greater, thus increasing the tolerance against outliers, where
                lower values decreases non-outlier interval and therefore crea-
                tes less tolerance against outliers.
        """
        v_min, q_1, q_3, v_max = np.percentile(N, (0, 25, 75, 100), axis=0)

        whis_iqr = whis * (q_3 - q_1)

        cut_low = q_1 - whis_iqr
        cut_high = q_3 + whis_iqr

        return sum(np.logical_or(cut_low > v_min, cut_high < v_max))

    @classmethod
    def ft_range(cls, N: np.ndarray) -> np.ndarray:
        """Compute range (max - min) of each attribute."""
        return np.ptp(N, axis=0)

    @classmethod
    def ft_sd(cls, N: np.ndarray, ddof: int = 1) -> np.ndarray:
        """Compute standard deviation of each attribute.

        Args:
            ddof (:obj:`float`): degrees of freedom for standard
                deviation.
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
                    classes: t.Optional[np.ndarray] = None,
                    class_freqs: t.Optional[np.ndarray] = None) -> float:
        """Statistic test for homogeneity of covariances.

        Args:
            epsilon (:obj:`float`): a tiny value to prevent division by zero.
        """
        num_inst, num_col = N.shape

        if classes is None or class_freqs is None:
            classes, class_freqs = np.unique(y, return_counts=True)

        num_classes = classes.size

        sample_cov_matrices = np.array(
            [np.cov(N[y == cl, :], rowvar=False) for cl in classes])

        vec_weight = class_freqs - 1.0 + epsilon

        pooled_cov_mat = np.array([
            weight * S_i
            for weight, S_i in zip(vec_weight, sample_cov_matrices)
        ]).sum(axis=0) / (num_inst - num_classes)

        gamma = 1.0 - (
            (2.0 * num_col**2.0 + 3.0 * num_col - 1.0) /
            (epsilon + 6.0 * (num_col + 1.0) *
             (num_classes - 1.0))) * (sum(1.0 / vec_weight) - 1.0 /
                                      (epsilon + num_inst - num_classes))

        try:
            vec_logdet = [
                np.math.log(epsilon + np.linalg.det(S_i))
                for S_i in sample_cov_matrices
            ]

            m_factor = (gamma * ((num_inst - num_classes) * np.math.log(
                np.linalg.det(pooled_cov_mat)) - np.dot(
                    vec_weight, vec_logdet)))

        except np.linalg.LinAlgError:
            return np.nan

        return np.exp(
            m_factor / (epsilon + num_col * (num_inst - num_classes)))

    @classmethod
    def ft_skewness(cls, N: np.ndarray, method: int = 3,
                    bias: bool = True) -> np.ndarray:
        """Compute skewness for each attribute.

        Args:
            bias (:obj:`bool`): If False, then the calculations are
                corrected for statistical bias.


        method (:obj:`int`, optional): defines the strategy used for
            estimate data skewness. Used for total compatibility with
            R package ``e1071``. The options must be one of the follo-
            wing:

            Option      Formula
            -------------------
            1           Skew_1 = m_3 / m_2^(3/2) (default of ``scipy.stats``)
            2           Skew_2 = Skew_1 * sqrt(n(n-1)) / (n-2)
            3           Skew_3 = m_3 / s^3 = Skew_1 ((n-1)/n)^(3/2)

            Where ``n`` is the number of elements in ``values`` and
            m_i is the ith momentum of ``values``.

            Note that if the selected method is unable to be calculated due
            to division by zero, then the first method will be used instead.
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
                    N: np.ndarray,
                    normalize: bool = True,
                    epsilon: float = 1.0e-8) -> np.ndarray:
        """Compute (normalized) sparsity metric for each attribute.

        Sparcity S of a vector x of numeric values is defined as

            S(x) = (1.0 / (n - 1.0)) * ((n / phi(x)) - 1.0),

        where
            - n is the number of instances in dataset N.
            - phi(x) is the number of distinct values in x.

        Args:
            normalize (:obj:`bool`): if True, then the output will be
                S(x) as calculated above. Otherwise, output will not
                be multiplied by (1.0 / (n - 1.0)) factor (i.e. new
                output is S'(x) = ((n / phi(x)) - 1.0)).

            epsilon (:obj:`float`): a tiny value to prevent division by zero.
        """

        ans = np.array([attr.size / np.unique(attr).size for attr in N.T])

        num_inst, _ = N.shape

        norm_factor = 1.0
        if normalize:
            norm_factor = 1.0 / (epsilon + num_inst - 1.0)

        return (ans - 1.0) * norm_factor

    @classmethod
    def ft_t_mean(cls, N: np.ndarray, pcut: float = 0.2) -> np.ndarray:
        """Compute trimmed mean of each attribute.

        Args:
            pcut (:obj:`float`): percentage of cut from both ``lower``
                and ``higher`` values. This value should be in inter-
                val [0.0, 0.5), where if 0.0 the return value is the
                default mean calculation. If pcut < 0.0, then
                :obj:`np.nan` will be returned.
        """
        if pcut < 0:
            return np.array([np.nan])

        return scipy.stats.trim_mean(N, proportiontocut=pcut)

    @classmethod
    def ft_var(cls, N: np.ndarray, ddof: int = 1) -> np.ndarray:
        """Compute variance of each attribute.

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
                    eig_vals: t.Optional[np.ndarray] = None,
                    classes: t.Optional[np.ndarray] = None,
                    class_freqs: t.Optional[np.ndarray] = None) -> float:
        """Compute Wilks' Lambda value.

        The Wilk's Lambda L is calculated as:

            L = prod(1.0 / (1.0 + lda_eig_i))

        Where ``lda_eig_i`` is the ith eigenvalue of Fisher's Linear Discri-
        minant Analysis Matrix. Check ``ft_can_cor`` documentation for more
        in-depth information about this value.
        """
        if eig_vals is None:
            if classes is None or class_freqs is None:
                classes, class_freqs = np.unique(y, return_counts=True)

            eig_vals, _ = MFEStatistical._linear_disc_mat_eig(
                N, y, classes=classes, class_freqs=class_freqs)

            _, num_attr = N.shape

            eig_vals = MFEStatistical._filter_eig_vals(
                eig_vals=eig_vals,
                num_attr=num_attr,
                num_classes=classes.size)

        if not isinstance(eig_vals, np.ndarray):
            eig_vals = np.array(eig_vals)

        if eig_vals.size == 0:
            return np.nan

        return np.prod(1.0 / (1.0 + eig_vals))
