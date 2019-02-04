"""Module dedicated to extraction of Statistical Metafeatures.

Notes:
    For more information about the metafeatures implemented here,
    check out `Rivolli et al.`_.

References:
    .. _Rivolli et al.:
        "Towards Reproducible Empirical Research in Meta-Learning",
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
            a single value or a generic Sequence (preferably a np.ndarray)
            type with numeric values.
    """

    @classmethod
    def _linear_disc_mat_eig(cls, N: np.ndarray,
                             y: np.ndarray,
                             ) -> t.Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvals/vecs of Fisher's Linear Discriminant Analysis.

        More specificaly, the eigenvalues and eigenvectors are calculated
        from matrix S = (Scatter_Within_Mat)^(-1) * (Scatter_Between_Mat).

        Check ``ft_can_cor`` documentation for more in-depth information
        about this matrix.

        Return:
            tuple(np.ndarray, np.ndarray): eigenvalues and eigenvectors
                (in this order) of Fisher's Linear Discriminant Analysis
                Matrix.
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

        class_val_freq = np.unique(y, return_counts=True)

        N = N.astype(float)

        scatter_within = compute_scatter_within(N, y, class_val_freq)
        scatter_between = compute_scatter_between(N, y, class_val_freq)

        try:
            scatter_within_inv = np.linalg.inv(scatter_within)

            return np.linalg.eig(np.matmul(scatter_within_inv,
                                           scatter_between))

        except (np.linalg.LinAlgError, ValueError):
            return np.array([np.nan]), np.array([np.nan])

    @classmethod
    def _filter_eig_vals(cls,
                         eig_vals: np.ndarray,
                         data: np.ndarray,
                         y: np.ndarray,
                         eig_vecs: t.Optional[np.ndarray] = None,
                         filter_imaginary: bool = True,
                         filter_less_relevant: float = True,
                         epsilon: float = 1.0e-8,
                         ) -> np.ndarray:
        """Get most expressive eigenvalues (higher absolute value).

        This function returns N eigenvalues, such that:

            N < min(num_class, num_attr)

        Args:
            filter_imaginary (:obj:`bool`, optional): if True, remove ima-
                ginary valued eigenvalues and its correspondent eigenvect-
                ors.

            filter_less_relevant (:obj:`bool`, optional): if True, remove
                eigenvalues smaller than ``epsilon``.

            epsilon (:obj:`float`, optional): a very small value used to
                determine ``less relevant`` eigenvalues.
        """
        _, num_cols = data.shape
        num_classes = np.unique(y).size
        max_valid_eig = min(num_cols, num_classes)

        if eig_vals.size <= max_valid_eig:
            if eig_vecs:
                return eig_vals, eig_vecs

            return eig_vals

        eig_vals = np.array(sorted(
            eig_vals, key=abs, reverse=True)[:max_valid_eig])

        if not filter_imaginary and not filter_less_relevant:
            if eig_vecs:
                return eig_vals, eig_vecs

            return eig_vals

        indexes_to_keep = np.array(eig_vals.size * [True])

        if filter_imaginary:
            indexes_to_keep = np.logical_and(np.isreal(eig_vals),
                                             indexes_to_keep)

        if filter_less_relevant:
            indexes_to_keep = np.logical_and(abs(eig_vals) > epsilon,
                                             indexes_to_keep)

        eig_vals = eig_vals[indexes_to_keep]

        if eig_vecs:
            eig_vecs = eig_vecs[indexes_to_keep]

        if filter_imaginary:
            eig_vals = eig_vals.real

        if eig_vecs:
            return eig_vals, eig_vecs

        return eig_vals

    @classmethod
    def ft_can_cor(cls, N: np.ndarray, y: np.ndarray,
                   epsilon: float = 1.0e-10) -> np.ndarray:
        """Compute canonical correlations of data.

        The canonical correlations p are defined as shown below:

            p_i = sqrt(lda_eig_i / (1.0 + lda_eig_i))

        Where ``lda_eig_i`` is the ith eigenvalue of Fisher's Linear Discri-
        minant Analysis Matrix S defined as:

            S = (Scatter_Within_Mat)^(-1) * (Scatter_Between_Mat),

        where
            Scatter_Within_Mat = sum((N_c - 1.0) * Covariance(X_c)),
            N_c is the number of instances of class c and X_c are the
            instances of class c. Effectively, this is exactly just the
            summation of all Covariance matrices between instances of the
            same class without dividing then by the number of instances.

            Scatter_Between_Mat = sum(N_c * (U_c - U) * (U_c - U)^T), N_c
            is the number of instances of class c, U_c is the mean coordi-
            nates of instances of class c, and U is the mean value of coor-
            dinates of all instances in the dataset.

        Args:
            epsilon (:obj:`float`): a very small value to prevent division by
                zero.
        """
        eig_vals, _ = MFEStatistical._linear_disc_mat_eig(N, y)

        eig_vals = MFEStatistical._filter_eig_vals(eig_vals=eig_vals,
                                                   data=N, y=y)

        return (eig_vals / (epsilon + 1.0 + eig_vals))**0.5

    @classmethod
    def ft_gravity(cls,
                   N: np.ndarray,
                   y: np.ndarray,
                   norm_ord: t.Union[int, float] = 2) -> float:
        """Computes distance between minority and majority classes center of mass.

        The center of mass of a class is the average value of each attribute
        between instances of the same class.

        The majority and minority classes can not be the same, even if
        all classes have the same number of instances.

        Args:
            norm_ord (:obj:`numeric`): Minkowski distance parameter. Minkowski
            distance has the following popular cases for this argument value:

                norm_ord    Distance name
                -------------------------
                -inf        Min value
                1           Manhattan/cityblock
                2           Euclidean
                +inf        Max value (infinite norm)

        Raises:
            ValueError: if ``norm_ord`` is not numeric.
        """
        classes, freqs = np.unique(y, return_counts=True)

        class_freq_most, _ = max(zip(classes, freqs), key=lambda x: x[1])

        class_freq_most_ind = np.where(class_freq_most == classes)[0]

        classes = np.delete(classes, class_freq_most_ind)
        freqs = np.delete(freqs, class_freq_most_ind)

        class_freq_least, _ = min(zip(classes, freqs), key=lambda x: x[1])

        center_freq_class_most = N[y == class_freq_most, :].mean(axis=0)
        center_freq_class_least = N[y == class_freq_least, :].mean(axis=0)

        return np.linalg.norm(
            center_freq_class_most - center_freq_class_least, ord=norm_ord)

    @classmethod
    def ft_cor(cls, N: np.ndarray) -> np.ndarray:
        """Absolute value of correlation between distinct column pairs."""
        corr_mat = np.corrcoef(N, rowvar=False)

        if not isinstance(corr_mat, np.ndarray) and np.isnan(corr_mat):
            return np.array([np.nan])

        res_num_rows, _ = corr_mat.shape

        inf_triang_vals = corr_mat[np.tril_indices(res_num_rows, k=-1)]

        return abs(inf_triang_vals)

    @classmethod
    def ft_cov(cls, N: np.ndarray) -> np.ndarray:
        """Absolute value of covariance between distinct column pairs."""
        cov_mat = np.cov(N, rowvar=False)

        res_num_rows, _ = cov_mat.shape

        inf_triang_vals = cov_mat[np.tril_indices(res_num_rows, k=-1)]

        return abs(inf_triang_vals)

    @classmethod
    def ft_nr_disc(cls, N: np.ndarray, y: np.ndarray) -> float:
        """Compute number of canonical corr. between each attr. and class.

        This is effectively the size of return value of ``ft_can_cor`` method.
        """
        can_cor = MFEStatistical.ft_can_cor(N, y)

        if isinstance(can_cor, np.ndarray):
            return can_cor.size

        return np.nan

    @classmethod
    def ft_eigenvalues(cls, N: np.ndarray) -> np.ndarray:
        """Returns eigenvalues of covariance matrix of N attributes."""
        cov_mat = np.cov(N, rowvar=False)

        try:
            eigvals = np.linalg.eigvals(cov_mat)

        except (np.linalg.LinAlgError, ValueError):
            return np.array([np.nan])

        return eigvals

    @classmethod
    def ft_g_mean(cls, N: np.ndarray, allow_zeros: bool = False,
                  epsilon: float = 1.0e-10) -> np.ndarray:
        """Geometric mean of each column.

        Args:
            allow_zeros (:obj:`bool`): if True, than all attributes with zero
                values will have geometric mean set to zero. Otherwise, their
                geometric mean are set to :obj:`np.nan`.

            epsilon (:obj:`float`): a very small value which all values with
                absolute value lesser than it are considered zero-valued.
        """
        min_values = N.min(axis=0)

        if allow_zeros:
            cols_invalid = min_values < 0.0
            cols_zero = 0.0 <= abs(min_values) < epsilon
            cols_valid = np.logical_not(np.logical_or(cols_invalid,
                                                      cols_zero))

        else:
            cols_invalid = min_values <= epsilon
            cols_valid = np.logical_not(cols_invalid)

        _, num_col = N.shape
        g_mean = np.zeros(num_col)

        g_mean[cols_valid] = scipy.stats.mstats.gmean(
            N[:, cols_valid], axis=0)

        g_mean[cols_invalid] = np.nan

        return g_mean

    @classmethod
    def ft_h_mean(cls, N: np.ndarray, epsilon: float = 1.0e-8) -> np.ndarray:
        """Harmonic mean of each column."""
        return scipy.stats.mstats.hmean(N + epsilon, axis=0)

    @classmethod
    def ft_iq_range(cls, N: np.ndarray) -> np.ndarray:
        """Compute Interquartile Range (IQR) of each column."""
        return scipy.stats.iqr(N, axis=0)

    @classmethod
    def ft_kurtosis(cls, N: np.ndarray, method: int = 3,
                    bias: bool = True) -> np.ndarray:
        """Compute Kurtosis of each attribute of N.

        Args:
            bias (:obj:`bool`): If False, then the calculations are corrected
                for statistical bias.

        method (:obj:`int`, optional): defines the strategy used for
            estimate data kurtosis. Used for total compatibility with
            R package ``e1071``. The options must be one of the follo-
            wing:

            Option      Formula
            -------------------
            1           Kurt_1 = m_4 / m_2^2 - 3. (default of ``scipy.stats``)
            2           Kurt_2 = ((n+1) * Kurt_1 + 6) * (n-1) / ((n-2)*(n-3)).
            3           Kurt_3 = m_4 / s^4 - 3 = (Kurt_1+3) * (1 - 1/n)^2 - 3.

            Where ``n`` is the number of elements in ``values`` and
            m_i is the ith momentum of ``values``.

            Note that if the selected method is unable to be calculated due
            to division by zero, then the first method will be used instead.
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
        """Computes Median Absolute Deviation (MAD) adjusted by a factor.

        Args:
            factor (:obj:`float`): multiplication factor for output correction.
            The default ``factor`` is 1.4826 due to fact that it is an appro-
            ximated result of MAD of a normally distributed data, so it make
            this method result comparable with this sort of data.
         """
        median_dev = abs(N - np.median(N, axis=0))
        return np.median(median_dev, axis=0) * factor

    @classmethod
    def ft_max(cls, N: np.ndarray) -> np.ndarray:
        """Get maximum value from each N attribute."""
        return N.max(axis=0)

    @classmethod
    def ft_mean(cls, N: np.ndarray) -> np.ndarray:
        """Returns the mean value of each data column."""
        return N.mean(axis=0)

    @classmethod
    def ft_median(cls, N: np.ndarray) -> np.ndarray:
        """Get median value from each N attribute."""
        return np.median(N, axis=0)

    @classmethod
    def ft_min(cls, N: np.ndarray) -> np.ndarray:
        """Get minimum value from each N attribute."""
        return N.min(axis=0)

    @classmethod
    def ft_nr_cor_attr(cls,
                       N: np.ndarray,
                       threshold: float = 0.5,
                       normalize: bool = True,
                       epsilon: float = 1.0e-8) -> t.Union[int, float]:
        """Number of attribute pairs with corr. equal or greater than a threshold.

        Args:
            threshold (:obj:`float`): value of threshold, where correlation is
                assumed to be strong if its absolute value is equal or greater
                than it.

            normalize (:obj:`bool`): if True, the result will be normalized by
                a factor of 2 / (d * (d - 1)), where d = number of attributes
                (columns) in N.

            epsilon (:obj:`float`): a very small value to prevent division by
                zero.
        """
        abs_corr_vals = MFEStatistical.ft_cor(N)

        if not isinstance(abs_corr_vals, np.ndarray):
            abs_corr_vals = np.array(abs_corr_vals)

        _, num_attr = N.shape

        norm_factor = 1

        if normalize:
            norm_factor = 2.0 / (epsilon + num_attr * (num_attr - 1.0))

        return sum(abs_corr_vals >= threshold) * norm_factor

    @classmethod
    def ft_nr_norm(cls, N: np.ndarray, threshold: float = 0.1) -> int:
        """Number of attr. with normal distribution based in Shapiro-Wilk test.

        Args:
            threshold (:obj:`float`): threshold to consider the p-value of
                Shapiro-Wilk test of each attribute small enough to assume
                normal distribution.
        """
        nr_norm = 0

        for attr in N.T:
            _, p_value = scipy.stats.shapiro(attr)
            nr_norm += p_value < threshold

        return nr_norm

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
    def ft_sd_ratio(cls, N: np.ndarray, y: np.ndarray,
                    epsilon: float = 1.0e-8) -> float:
        """Statistic test for homogeneity of covariances.

        Args:
            epsilon (:obj:`float`): a very small value to prevent division by
                zero.
        """
        num_inst, num_col = N.shape
        classes, classes_freqs = np.unique(y, return_counts=True)
        num_classes = classes.size

        sample_cov_matrices = np.array(
            [np.cov(N[y == cl, :], rowvar=False) for cl in classes])

        vec_weight = classes_freqs - 1.0 + epsilon

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

            epsilon (:obj:`float`): a very small value to prevent division by
                zero.
        """

        ans = np.array([attr.size / np.unique(attr).size for attr in N.T])

        num_inst, _ = N.shape

        norm_factor = 1.0
        if normalize:
            norm_factor = 1.0 / (epsilon + num_inst - 1.0)

        return (ans - 1.0) * norm_factor

    @classmethod
    def ft_t_mean(cls, N: np.ndarray,
                  pcut: float = 0.2) -> np.ndarray:
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
    def ft_w_lambda(cls, N: np.ndarray, y: np.ndarray) -> float:
        """Compute Wilks' Lambda value.

        The Wilk's Lambda L is calculated as:

            L = prod(1.0 / (1.0 + lda_eig_i))

        Where ``lda_eig_i`` is the ith eigenvalue of Fisher's Linear Discri-
        minant Analysis Matrix. Check ``ft_can_cor`` documentation for more
        in-depth information about this value.
        """
        eig_vals, _ = MFEStatistical._linear_disc_mat_eig(N, y)

        eig_vals = MFEStatistical._filter_eig_vals(eig_vals=eig_vals,
                                                   data=N, y=y)

        if eig_vals.size == 0:
            return np.nan

        return np.prod(1.0 / (1.0 + eig_vals))
