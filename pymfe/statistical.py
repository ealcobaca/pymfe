"""A module dedicated to the extraction of statistical metafeatures."""
import typing as t
import warnings

import numpy as np
import scipy
import sklearn.preprocessing
import sklearn.cross_decomposition

import pymfe._summary as _summary


class MFEStatistical:
    """Keep methods for metafeatures of ``Statistical`` group.

    The convention adopted for metafeature-extraction related methods
    is to always start with ``ft_`` prefix in order to allow automatic
    method detection. This prefix is predefined within ``_internal``
    module.

    All method signature follows the conventions and restrictions listed
    below:

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
       arguments must be strictly optional (i.e., has a predefined
       default value).

    3. It is assumed that the user can change any optional argument, without
       any previous verification for both type or value, via kwargs argument of
       ``extract`` method of MFE class.

    4. The return value of all feature-extraction methods should be a single
       value or a generic List (preferably an np.ndarray)
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
    def precompute_statistical_class(
        cls, y: t.Optional[np.ndarray] = None, **kwargs
    ) -> t.Dict[str, t.Any]:
        """Precompute distinct classes and its abs. frequencies from ``y``.

        Parameters
        ----------
        y : :obj:`np.ndarray`
            The target attribute from fitted data.

        kwargs:
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
        :obj:`dict`
            With following precomputed items:
                * ``classes`` (:obj:`np.ndarray`): distinct classes of ``y``,
                  if ``y`` is not :obj:`NoneType`.
                * ``class_freqs`` (:obj:`np.ndarray`): absolute class
                  frequencies of ``y``, if ``y`` is not :obj:`NoneType`.
        """
        precomp_vals = {}  # type: t.Dict[str, t.Any]

        if y is not None and not {"classes", "class_freqs"}.issubset(kwargs):
            classes, class_freqs = np.unique(y, return_counts=True)

            precomp_vals["classes"] = classes
            precomp_vals["class_freqs"] = class_freqs

        return precomp_vals

    @classmethod
    def precompute_can_cors(
        cls,
        N: t.Optional[np.ndarray] = None,
        y: t.Optional[np.ndarray] = None,
        **kwargs
    ) -> t.Dict[str, t.Any]:
        """Precompute canonical correlations and its eigenvalues.

        Parameters
        ----------
        N : :obj:`np.ndarray`, optional
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        kwargs:
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
            :obj:`dict`
                With following precomputed items:
                    - ``can_cors`` (:obj:`np.ndarray`): canonical correlations
                      between ``N`` and the one-hot encoded version of ``y``.
                    - ``can_cor_eigvals`` (:obj:`np.ndarray`): eigenvalues
                      related to the canonical correlations.
        """
        precomp_vals = {}  # type: t.Dict[str, t.Any]

        if (
            y is not None
            and N is not None
            and N.size
            and not {"can_cors", "can_cor_eigvals"}.issubset(kwargs)
        ):
            can_cors = cls._calc_can_cors(N=N, y=y)

            precomp_vals["can_cors"] = can_cors
            precomp_vals["can_cor_eigvals"] = cls._can_cor_to_eigval(can_cors)

        return precomp_vals

    @classmethod
    def precompute_statistical_cor_cov(
        cls, N: t.Optional[np.ndarray] = None, ddof: int = 1, **kwargs
    ) -> t.Dict[str, t.Any]:
        """Precomputes the correlation and covariance matrix of numerical data.

        Be cautious in allowing this precomputation method on huge datasets, as
        this precomputation method may be very memory hungry.

        Parameters
        ----------
        N : :obj:`np.ndarray`, optional
            Numerical fitted data.

        ddof : int, optional
            Degrees of freedom of covariance matrix.

        kwargs:
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
            :obj:`dict`
                With following precomputed items:
                    - ``cov_mat`` (:obj:`np.ndarray`): covariance matrix of
                      ``N``, if ``N`` is not :obj:`NoneType`.
                    - ``abs_corr_mat`` (:obj:`np.ndarray`): absolute
                      correlation matrix of ``N``, if ``N`` is not
                      :obj:`NoneType`.
        """
        precomp_vals = {}  # type: t.Dict[str, t.Any]

        if N is not None and N.size:
            if "cov_mat" not in kwargs:
                precomp_vals["cov_mat"] = np.cov(N, rowvar=False, ddof=ddof)

            if "abs_corr_mat" not in kwargs:
                abs_corr_mat = np.abs(np.corrcoef(N, rowvar=False))

                if not isinstance(abs_corr_mat, np.ndarray) and np.isnan(
                    abs_corr_mat
                ):
                    abs_corr_mat = np.array([np.nan])

                precomp_vals["abs_corr_mat"] = abs_corr_mat

        return precomp_vals

    @staticmethod
    def _can_cor_to_eigval(can_cors: np.ndarray) -> np.ndarray:
        """Transform canonical correlations into corresponding eigenvalues.

        The transformation uses the following relationship:

            can_cor_i = sqrt(can_cor_eigval_i / (1 + can_cor_eigval_i))

        Or, equivalently:

            can_cor_eigval_i = can_cor_i**2 / (1 - can_cor_i**2)

        So, the option to return the eigenvalues is meant to simplify
        code that uses those values, not to generate extra inforation.
        """
        sqr_can_cors = np.square(can_cors)
        can_cor_eig_vals = sqr_can_cors / (1 - sqr_can_cors)
        return can_cor_eig_vals

    @classmethod
    def _calc_can_cors(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        max_iter: int = 500,
    ) -> np.ndarray:
        """Calculate the Canonical Correlations between ``N`` and ``y.``

        Note that the canonical correlations are calculated using the
        one-hot encoded version of ``y.``

        At most min(num_classes, num_attr) canonical correlations are
        kept.
        """
        y_bin = sklearn.preprocessing.OneHotEncoder(
            sparse=False,
            drop="first",
        ).fit_transform(y.reshape(-1, 1))

        num_classes, num_attr = y_bin.shape[1], N.shape[1]
        # Note: 'n_components' is a theoretical upper bound, so it is not
        # guaranteed that exactly 'n_components' will be returned.
        n_components = min(num_classes, num_attr)

        # Note: 'sklearn.cross_decomposition.CCA' issues UserWarnings
        # whenever less than 'n_components' are got. However, this is
        # already taken into account in this function, so no need for
        # those warnings.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            cca_model = sklearn.cross_decomposition.CCA(
                n_components=n_components, max_iter=max_iter
            )

            try:
                cca_model.fit(N, y_bin)

            except StopIteration:
                pass

            N_tf, y_tf = cca_model.transform(N, y_bin)

        ind = 0
        can_cors = np.zeros(n_components, dtype=float)

        while ind < n_components and np.any(np.flatnonzero(N_tf[:, ind])):
            can_cors[ind] = np.corrcoef(N_tf[:, ind], y_tf[:, ind])[0, 1]
            ind += 1

        can_cors = can_cors[:ind]

        return can_cors

    @classmethod
    def ft_can_cor(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        can_cors: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute canonical correlations of data.

        The canonical correlations are calculated between the attributes
        in ``N`` and the binarized (one-hot encoded) version of ``y``.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        y : :obj:`np.ndarray`
            Target attribute.

        can_cors : :obj:`np.ndarray`, optional
            Canonical correlations between ``N`` and the one-hot encoded
            version of ``y``. Argument used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Canonical correlations of the data.

        References
        ----------
        .. [1] Alexandros Kalousis. Algorithm Selection via Meta-Learning.
           PhD thesis, Faculty of Science of the University of Geneva, 2002.
        """
        if can_cors is None:
            can_cors = cls._calc_can_cors(N=N, y=y)

        return can_cors

    @classmethod
    def ft_gravity(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        norm_ord: t.Union[int, float] = 2,
        classes: t.Optional[np.ndarray] = None,
        class_freqs: t.Optional[np.ndarray] = None,
        cls_inds: t.Optional[np.ndarray] = None,
    ) -> float:
        """Compute the distance between minority and majority classes center
        of mass.

        The center of mass of a class is the average value of each attribute
        between instances of the same class.

        The majority and minority classes cannot be the same, even if every
        class has the same number of instances.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        y : :obj:`np.ndarray`
            Target attribute.

        norm_ord : :obj:`numeric`, optional
            Minkowski Distance parameter. Minkowski Distance has the following
            popular cases for this argument value

            +-----------+---------------------------+
            |norm_ord   | Distance name             |
            +-----------+---------------------------+
            |-> -inf    | Min value                 |
            +-----------+---------------------------+
            |1.0        | Manhattan/City Block      |
            +-----------+---------------------------+
            |2.0        | Euclidean                 |
            +-----------+---------------------------+
            |-> +inf    | Max value (infinite norm) |
            +-----------+---------------------------+

        classes : :obj:`np.ndarray`, optional
            Distinct classes of ``y``.

        class_freqs : :obj:`np.ndarray`, optional
            Absolute frequencies of each distinct class in target attribute
            ``y`` or ``classes``. If ``classes`` is given, then this argument
            must be paired with it by index.

        cls_inds : :obj:`np.ndarray`, optional
            Boolean array which indicates the examples of each class.
            The rows represents each distinct class, and the columns
            represents the instances. Used to take advantage of
            precomputations.

        Returns
        -------
        float
            Gravity of the numeric dataset.

        Raises
        ------
        :obj:`ValueError`
            If ``norm_ord`` is not numeric.

        References
        ----------
        .. [1] Shawkat Ali and Kate A. Smith. On learning algorithm
           selection for classification. Applied Soft Computing,
           6(2):119 – 138, 2006.
        """
        if classes is None or class_freqs is None:
            classes, class_freqs = np.unique(y, return_counts=True)

        _classes = np.asarray(classes)  # type: np.ndarray
        _class_freqs = np.asarray(class_freqs, dtype=int)  # type: np.ndarray

        ind_cls_maj = np.argmax(_class_freqs)
        class_maj = _classes[ind_cls_maj]

        _classes = np.delete(_classes, ind_cls_maj)
        _class_freqs = np.delete(_class_freqs, ind_cls_maj)

        ind_cls_min = np.argmin(_class_freqs)

        if cls_inds is not None:
            insts_cls_maj = N[cls_inds[ind_cls_maj, :], :]
            # Adjusting minoritary class index due to 'delete' operation
            ind_cls_min += ind_cls_min >= ind_cls_maj
            insts_cls_min = N[cls_inds[ind_cls_min, :], :]

        else:
            class_min = _classes[ind_cls_min]
            insts_cls_maj = N[y == class_maj, :]
            insts_cls_min = N[y == class_min, :]

        gravity = np.linalg.norm(
            insts_cls_maj.mean(axis=0) - insts_cls_min.mean(axis=0),
            ord=norm_ord,
        )

        return gravity

    @classmethod
    def ft_cor(
        cls, N: np.ndarray, abs_corr_mat: t.Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute the absolute value of the correlation of distinct dataset
        column pairs.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        abs_corr_mat : :obj:`np.ndarray`, optional
            Absolute correlation matrix of ``N``. Argument used to exploit
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Absolute value of correlation between distinct attributes.

        References
        ----------
        .. [1] Ciro Castiello, Giovanna Castellano, and Anna Maria Fanelli.
           Meta-data: Characterization of input features for meta-learning.
           In 2nd International Conference on Modeling Decisions for
           Artificial Intelligence (MDAI), pages 457–468, 2005.
        .. [2] Matthias Reif, Faisal Shafait, Markus Goldstein, Thomas Breuel,
           and Andreas Dengel. Automatic classifier selection for non-experts.
           Pattern Analysis and Applications, 17(1):83–96, 2014.
        .. [3] Donald Michie, David J. Spiegelhalter, Charles C. Taylor, and
           John Campbell. Machine Learning, Neural and Statistical
           Classification, volume 37. Ellis Horwood Upper Saddle River, 1994.
        """
        if abs_corr_mat is None:
            abs_corr_mat = np.abs(np.corrcoef(N, rowvar=False))

        res_num_rows, _ = abs_corr_mat.shape

        inf_triang_vals = abs_corr_mat[np.tril_indices(res_num_rows, k=-1)]

        return np.abs(inf_triang_vals)

    @classmethod
    def ft_cov(
        cls,
        N: np.ndarray,
        ddof: int = 1,
        cov_mat: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the absolute value of the covariance of distinct dataset
        attribute pairs.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        ddof : int, optional
            Degrees of freedom for covariance matrix.

        cov_mat : :obj:`np.ndarray`, optional
            Covariance matrix of ``N``. Argument meant to exploit
            precomputations. Note that this argument value is not the same as
            this method return value, as it only returns the lower-triangle
            values from ``cov_mat``.

        Returns
        -------
        :obj:`np.ndarray`
            Absolute value of covariances between distinct attributes.

        References
        ----------
        .. [1] Ciro Castiello, Giovanna Castellano, and Anna Maria Fanelli.
           Meta-data: Characterization of input features for meta-learning.
           In 2nd International Conference on Modeling Decisions for
           Artificial Intelligence (MDAI), pages 457–468, 2005.
        .. [2] Donald Michie, David J. Spiegelhalter, Charles C. Taylor, and
           John Campbell. Machine Learning, Neural and Statistical
           Classification, volume 37. Ellis Horwood Upper Saddle River, 1994.
        """
        if cov_mat is None:
            cov_mat = np.cov(N, rowvar=False, ddof=ddof)

        res_num_rows, _ = cov_mat.shape

        inf_triang_vals = cov_mat[np.tril_indices(res_num_rows, k=-1)]

        return np.abs(inf_triang_vals)

    @classmethod
    def ft_nr_disc(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        can_cors: t.Optional[np.ndarray] = None,
    ) -> t.Union[int, float]:
        """Compute the number of canonical correlation between each attribute
        and class.

        This method return value is effectively the size of the return value
        of ``ft_can_cor`` method. Check its documentation for more in-depth
        details.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        y : :obj:`np.ndarray`
            Target attribute.

        can_cors : :obj:`np.ndarray`, optional
            Canonical correlations between ``N`` and the one-hot encoded
            version of ``y``. Argument used to take advantage of
            precomputations.

        Returns
        -------
        int or float
            Number of canonical correlations between each attribute and
            class, if ``ft_can_cor`` is executed successfully. Returns
            :obj:`np.nan` otherwise.

        References
        ----------
        .. [1] Guido Lindner and Rudi Studer. AST: Support for algorithm
           selection with a CBR approach. In European Conference on
           Principles of Data Mining and Knowledge Discovery (PKDD),
           pages 418 – 423, 1999.
        """
        if can_cors is None:
            can_cors = cls.ft_can_cor(N=N, y=y)

        return can_cors.size

    @classmethod
    def ft_eigenvalues(
        cls,
        N: np.ndarray,
        ddof: int = 1,
        cov_mat: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the eigenvalues of covariance matrix from dataset.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        ddof : int, optional
            Degrees of freedom for covariance matrix.

        cov_mat : :obj:`np.ndarray`, optional
            Covariance matrix of ``N``. Argument meant to exploit
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Eigenvalues of ``N`` covariance matrix.

        References
        ----------
        .. [1] Shawkat Ali and Kate A. Smith. On learning algorithm
           selection for classification. Applied Soft Computing,
           6(2):119 – 138, 2006.
        """
        if cov_mat is None:
            cov_mat = np.cov(N, rowvar=False, ddof=ddof)

        return np.linalg.eigvalsh(cov_mat)

    @classmethod
    def ft_g_mean(
        cls, N: np.ndarray, allow_zeros: bool = True, epsilon: float = 1.0e-10
    ) -> np.ndarray:
        """Compute the geometric mean of each attribute.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        allow_zeros : :obj:`bool`, optional
            If True, then the geometric mean of all attributes with zero values
            is set to zero. Otherwise, is set to :obj:`np.nan` these values.

        epsilon : float, optional
            A small value which all values with absolute value lesser than it
            is considered zero-valued. Used only if ``allow_zeros`` is False.

        Returns
        -------
        :obj:`np.ndarray`
            Attribute geometric means.

        References
        ----------
        .. [1] Shawkat Ali and Kate A. Smith-Miles. A meta-learning approach
           to automatic kernel selection for support vector machines.
           Neurocomputing, 70(1):173 – 186, 2006.
        """
        min_values = N.min(axis=0)

        if allow_zeros:
            cols_invalid = min_values < 0.0
            cols_zero = np.logical_and(min_values >= 0.0, min_values < epsilon)
            cols_valid = np.logical_not(np.logical_or(cols_invalid, cols_zero))

        else:
            cols_invalid = min_values <= epsilon
            cols_valid = np.logical_not(cols_invalid)

        _, num_col = N.shape
        g_mean = np.zeros(num_col, dtype=float)

        g_mean[cols_valid] = scipy.stats.gmean(N[:, cols_valid], axis=0)

        g_mean[cols_invalid] = np.nan

        # Note: the R MFE version can favors infinities over real values,
        # which is summarized as 'nan'. This version always tries to pick
        # a real value whenever it is available.
        return g_mean

    @classmethod
    def ft_h_mean(cls, N: np.ndarray) -> np.ndarray:
        """Compute the harmonic mean of each attribute.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        Returns
        -------
        :obj:`np.ndarray`
            Attribute harmonic means.

        References
        ----------
        .. [1] Shawkat Ali and Kate A. Smith-Miles. A meta-learning approach
           to automatic kernel selection for support vector machines.
           Neurocomputing, 70(1):173 – 186, 2006.
        """
        return scipy.stats.hmean(N, axis=0)

    @classmethod
    def ft_iq_range(cls, N: np.ndarray) -> np.ndarray:
        """Compute the interquartile range (IQR) of each attribute.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        Returns
        -------
        :obj:`np.ndarray`
            Attribute interquartile ranges.

        References
        ----------
        .. [1] Shawkat Ali and Kate A. Smith-Miles. A meta-learning approach
           to automatic kernel selection for support vector machines.
           Neurocomputing, 70(1):173 – 186, 2006.
        """
        return scipy.stats.iqr(N, axis=0)

    @classmethod
    def ft_kurtosis(
        cls, N: np.ndarray, method: int = 3, bias: bool = True
    ) -> np.ndarray:
        """Compute the kurtosis of each attribute.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        method : int, optional
            Defines the strategy used for estimate data kurtosis. Used for
            total compatibility with R package ``e1071``. The options must be
            one of the following:

            +--------+-----------------------------------------------+
            |Option  | Formula                                       |
            +--------+-----------------------------------------------+
            |1       | Kurt_1 = (m_4 / m_2**2 - 3)                   |
            |        | (default of `scipy.stats` package)            |
            +--------+-----------------------------------------------+
            |2       | Kurt_2 = (((n+1) * Kurt_1 + 6) * (n-1) / f_2),|
            |        | f_2 = ((n-2)*(n-3))                           |
            +--------+-----------------------------------------------+
            |3       | Kurt_3 = (m_4 / s**4 - 3)                     |
            |        |        = ((Kurt_1+3) * (1 - 1/n)**2 - 3)      |
            +--------+-----------------------------------------------+

            Where `n` is the number of instances in ``N``, `s` is the standard
            deviation of each attribute in ``N``, and `m_i` is the ith
            statistical momentum of each attribute in ``N``.

            Note that if the selected method is unable to be calculated due
            to division by zero, then the first method is used instead.

        bias : bool, optional
            If False, then the calculations are corrected for statistical bias.

        Returns
        -------
        :obj:`np.ndarray`
            Attribute kurtosis.

        References
        ----------
        .. [1] Donald Michie, David J. Spiegelhalter, Charles C. Taylor, and
           John Campbell. Machine Learning, Neural and Statistical
           Classification, volume 37. Ellis Horwood Upper Saddle River, 1994.
        """
        kurt_arr = np.apply_along_axis(
            func1d=_summary.sum_kurtosis,
            axis=0,
            arr=N,
            method=method,
            bias=bias,
        )

        return kurt_arr

    @classmethod
    def ft_mad(cls, N: np.ndarray, factor: float = 1.4826) -> np.ndarray:
        """Compute the Median Absolute Deviation (MAD) adjusted by a factor.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        factor : float, optional
            Multiplication factor for output correction. The default ``factor``
            is 1.4826 since it is an approximated result of MAD of a normally
            distributed data (with any mean and standard deviation of 1.0), so
            it makes this method result comparable with this sort of data.

        Returns
        -------
        :obj:`np.ndarray`
            Attribute MAD (Median Absolute Deviation.)

        References
        ----------
        .. [1] Shawkat Ali and Kate A. Smith. On learning algorithm
           selection for classification. Applied Soft Computing,
           6(2):119 – 138, 2006.
        """
        return scipy.stats.median_abs_deviation(
            x=N, axis=0, scale=1.0 / factor
        )

    @classmethod
    def ft_max(cls, N: np.ndarray) -> np.ndarray:
        """Compute the maximum value from each attribute.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        Returns
        -------
        :obj:`np.ndarray`
            Attribute maximum values.

        References
        ----------
        .. [1] Robert Engels and Christiane Theusinger. Using a data metric for
           preprocessing advice for data mining applications. In 13th European
           Conference on on Artificial Intelligence (ECAI), pages 430 – 434,
           1998.
        """
        return np.asfarray(N.max(axis=0))

    @classmethod
    def ft_mean(cls, N: np.ndarray) -> np.ndarray:
        """Compute the mean value of each attribute.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        Returns
        -------
        :obj:`np.ndarray`
            Attribute mean values.

        References
        ----------
        .. [1] Robert Engels and Christiane Theusinger. Using a data metric for
           preprocessing advice for data mining applications. In 13th European
           Conference on on Artificial Intelligence (ECAI), pages 430 – 434,
           1998.
        """
        return np.asfarray(N.mean(axis=0))

    @classmethod
    def ft_median(cls, N: np.ndarray) -> np.ndarray:
        """Compute the median value from each attribute.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        Returns
        -------
        :obj:`np.ndarray`
            Attribute median values.

        References
        ----------
        .. [1] Robert Engels and Christiane Theusinger. Using a data metric for
           preprocessing advice for data mining applications. In 13th European
           Conference on on Artificial Intelligence (ECAI), pages 430 – 434,
           1998.
        """
        return np.asfarray(np.median(N, axis=0))

    @classmethod
    def ft_min(cls, N: np.ndarray) -> np.ndarray:
        """Compute the minimum value from each attribute.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        Returns
        -------
        :obj:`np.ndarray`
            Attribute minimum values.

        References
        ----------
        .. [1] Robert Engels and Christiane Theusinger. Using a data metric for
           preprocessing advice for data mining applications. In 13th European
           Conference on on Artificial Intelligence (ECAI), pages 430 – 434,
           1998.
        """
        return np.asfarray(N.min(axis=0))

    @classmethod
    def ft_nr_cor_attr(
        cls,
        N: np.ndarray,
        threshold: float = 0.5,
        normalize: bool = True,
        abs_corr_mat: t.Optional[np.ndarray] = None,
    ) -> t.Union[int, float]:
        """Compute the number of distinct highly correlated pair of attributes.

        A pair of attributes is considered highly correlated if the
        absolute value of its covariance is equal or larger than a
        given ``threshold``.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        threshold : float, optional
            A value of the threshold, where correlation is assumed to be strong
            if its absolute value is equal or greater than it.

        normalize : bool, optional
            If True, the result is normalized by a factor of 2/(d*(d-1)), where
            `d` is number of attributes (columns) in ``N``.

        abs_corr_mat : :obj:`np.ndarray`, optional
            Absolute correlation matrix of ``N``. Argument used to exploit
            precomputations.

        Returns
        -------
        int | float
            If ``normalize`` is False, this method returns the number of
            highly correlated pair of distinct attributes. Otherwise,
            return the proportion of highly correlated attributes.

        References
        ----------
        .. [1] Mostafa A. Salama, Aboul Ella Hassanien, and Kenneth Revett.
           Employment of neural network and rough set in meta-learning.
           Memetic Computing, 5(3):165 – 177, 2013.
        """
        abs_corr_vals = cls.ft_cor(N, abs_corr_mat=abs_corr_mat)

        _, num_attr = N.shape

        norm_factor = 1.0

        if normalize:
            norm_factor = 2.0 / (num_attr * (num_attr - 1.0))

        return np.sum(abs_corr_vals >= threshold) * norm_factor

    @classmethod
    def ft_nr_norm(
        cls,
        N: np.ndarray,
        method: str = "shapiro-wilk",
        threshold: float = 0.05,
        failure: str = "soft",
        max_samples: int = 5000,
    ) -> t.Union[float, int]:
        """Compute the number of attributes normally distributed based in a
        given method.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        method : str, optional
            Select the normality test to be executed. This argument must assume
            one of the options shown below:

            - shapiro-wilk: from `scipy.stats.shapiro` documentation: the
              Shapiro-Wilk test tests the null hypothesis that the data was
              drawn from a normal distribution.

            - dagostino-pearson: from `scipy.stats.normaltest` documentation:
              It is based on D'Agostino and Pearson's, test that combines skew
              and kurtosis to produce an omnibus test of normality.

            - anderson-darling: from `scipy.stats.anderson` documentation: The
              Anderson-Darling tests the null hypothesis that a sample is
              drawn from a population that follows a particular distribution.
              In this method context, that `particular distribution` is fixed
              in the normal/gaussian.

            - all: perform all tests cited above. To consider an attribute
              normaly distributed all test results are taken into account with
              equal weight. Check ``failure`` argument for more information.

        threshold : float, optional
            Level of significance used to reject the null hypothesis of
            normality tests.

        failure : str, optional
            Used only if ``method`` argument value is `all`. This argument
            must assumed one value between `soft` or `hard`. If `soft`, then if
            a single test have its null hypothesis (which all states the data
            follows a Guassian distribution) rejected for some attribute, then
            that attribute is already considered normally distributed. If value
            is `hard`, then is necessary the rejection of the null hypothesis
            of every single normality test to consider the attribute normally
            distributed.

        max_samples : int, optional
            Max samples used while performing the normality tests.
            Shapiro-Wilks test p-value may not be accurate when sample size is
            higher than 5000. Note that the instances are NOT shuffled before
            doing this cutoff. This means that the very first ``max_samples``
            instances of the dataset ``N`` will be considered in the
            statistical tests.

        Returns
        -------
        int
            The number of normally distributed attributes based on the
            ``method``. If ``max_samples`` is non-positive, :obj:`np.nan`
            is returned instead.

        Raises
        ------
        ValueError
            If ``method`` or ``failure`` is not a valid option.

        References
        ----------
        .. [1] Christian Kopf, Charles Taylor, and Jorg Keller. Meta-Analysis:
           From data characterisation for meta-learning to meta-regression. In
           PKDD Workshop on Data Mining, Decision Support, Meta-Learning and
           Inductive Logic Programming, pages 15 – 26, 2000.
        """
        accepted_tests = (
            "shapiro-wilk",
            "dagostino-pearson",
            "anderson-darling",
            "all",
        )

        if method not in accepted_tests:
            raise ValueError(
                "Unknown method {0}. Select one between {1}".format(
                    method, accepted_tests
                )
            )

        if failure not in ("hard", "soft"):
            raise ValueError(
                '"failure" argument must be either "soft" '
                'or "hard" (got "{}").'.format(failure)
            )

        if max_samples <= 0:
            return np.nan

        num_inst, num_attr = N.shape

        max_row_index = min(max_samples, num_inst)

        test_results = []

        if method in ("shapiro-wilk", "all"):
            _, p_values_shapiro = np.apply_along_axis(
                func1d=scipy.stats.shapiro, axis=0, arr=N[:max_row_index, :]
            )

            test_results.append(p_values_shapiro > threshold)

        if method in ("dagostino-pearson", "all"):
            _, p_values_dagostino = scipy.stats.normaltest(
                N[:max_row_index, :], axis=0
            )

            test_results.append(p_values_dagostino > threshold)

        if method in ("anderson-darling", "all"):
            anderson_stats = np.repeat(False, num_attr)

            for attr_ind, attr_vals in enumerate(N[:max_row_index, :].T):
                stat_value, crit_values, signif_levels = scipy.stats.anderson(
                    attr_vals, dist="norm"
                )

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

        return float(np.sum(attr_is_normal))

    @classmethod
    def ft_nr_outliers(cls, N: np.ndarray, whis: float = 1.5) -> int:
        """Compute the number of attributes with at least one outlier value.

        An attribute has outlier if some value is outside the closed interval
        [first_quartile - WHIS * IQR, third_quartile + WHIS * IQR], where IQR
        is the Interquartile Range (third_quartile - first_quartile), and WHIS
        value is typically `1.5`.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        whis : float, optional
            A factor to multiply IQR and set up non-outlier interval
            (as stated above). Higher values make the interval more
            significant, thus increasing the tolerance against outliers, where
            lower values decrease non-outlier interval and, therefore, creates
            less tolerance against possible outliers.

        Returns
        -------
        int
            Number of attributes with at least one outlier.

        References
        ----------
        .. [1] Christian Kopf and Ioannis Iglezakis. Combination of task
           description strategies and case base properties for meta-learning.
           In 2nd ECML/PKDD International Workshop on Integration and
           Collaboration Aspects of Data Mining, Decision Support and
           Meta-Learning(IDDM), pages 65 – 76, 2002.
        .. [2] Peter J. Rousseeuw and Mia Hubert. Robust statistics for
           outlier detection. Wiley Interdisciplinary Reviews: Data Mining
           and Knowledge Discovery, 1(1):73 – 79, 2011.
        """
        v_min, q_1, q_3, v_max = np.percentile(N, (0, 25, 75, 100), axis=0)

        whis_iqr = whis * (q_3 - q_1)

        cut_low = q_1 - whis_iqr
        cut_high = q_3 + whis_iqr

        return np.sum(np.logical_or(cut_low > v_min, cut_high < v_max))

    @classmethod
    def ft_range(cls, N: np.ndarray) -> np.ndarray:
        """Compute the range (max - min) of each attribute.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        Returns
        -------
        :obj:`np.ndarray`
            Attribute ranges.

        References
        ----------
        .. [1] Shawkat Ali and Kate A. Smith-Miles. A meta-learning approach
           to automatic kernel selection for support vector machines.
           Neurocomputing, 70(1):173 – 186, 2006.
        """
        return np.asfarray(np.ptp(N, axis=0))

    @classmethod
    def ft_sd(cls, N: np.ndarray, ddof: int = 1) -> np.ndarray:
        """Compute the standard deviation of each attribute.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        ddof : float, optional
            Degrees of freedom for standard deviation.

        Returns
        -------
        :obj:`np.ndarray`
            Attribute standard deviations.

        References
        ----------
        .. [1] Robert Engels and Christiane Theusinger. Using a data metric for
           preprocessing advice for data mining applications. In 13th European
           Conference on on Artificial Intelligence (ECAI), pages 430 – 434,
           1998.
        """
        return np.asfarray(N.std(axis=0, ddof=ddof))

    @classmethod
    def ft_sd_ratio(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        ddof: int = 1,
        classes: t.Optional[np.ndarray] = None,
        class_freqs: t.Optional[np.ndarray] = None,
    ) -> float:
        """Compute a statistical test for homogeneity of covariances.

        The test applied is the Box's M Test for equivalence of
        covariances.

        The null hypothesis of this test states that the covariance
        matrices of the instances of every class are equal.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        y : :obj:`np.ndarray`
            Target attribute.

        ddof : int, optional
            Degrees of freedom for covariance matrix, calculated during this
            test.

        classes : :obj:`np.ndarray`, optional
            All distinct classes in target attribute ``y``. Used to exploit
            precomputations.

        class_freqs : :obj:`np.ndarray`, optional
            Absolute frequencies of each distinct class in target attribute
            ``y`` or ``classes``. If ``classes`` is given, then this argument
            must be paired with it by index.

        Returns
        -------
        float
            Homogeneity of covariances test result.

        Notes
        -----
        For details about how this test is applied, check out `Rivolli
        et al.`_ (pag. 32).

        .. _Rivolli et al.:
            "Towards Reproducible Empirical Research in Meta-Learning,"
            Rivolli et al. URL: https://arxiv.org/abs/1808.10406

        References
        ----------
        .. [1] Donald Michie, David J. Spiegelhalter, Charles C. Taylor, and
           John Campbell. Machine Learning, Neural and Statistical
           Classification, volume 37. Ellis Horwood Upper Saddle River, 1994.

        """

        def calc_sample_cov_mat(N, y, ddof):
            """Calculate the Sample Covariance Matrix for each class."""
            sample_cov_matrices = np.array(
                [
                    np.cov(N[y == cl, :], rowvar=False, ddof=ddof)
                    for cl in classes
                ]
            )

            return np.flip(m=sample_cov_matrices, axis=(0, 1))

        def calc_pooled_cov_mat(
            sample_cov_matrices: np.ndarray,
            vec_weight: np.ndarray,
            num_inst: int,
            num_classes: int,
        ) -> np.ndarray:
            """Calculate the Pooled Covariance Matrix."""
            pooled_cov_mat = np.array(
                [
                    weight * S_i
                    for weight, S_i in zip(vec_weight, sample_cov_matrices)
                ]
            ).sum(axis=0) / (num_inst - num_classes)

            return np.asfarray(pooled_cov_mat)

        def calc_gamma_factor(num_col, num_classes, num_inst):
            """Calculate the gamma factor which adjust the output."""
            gamma = 1.0 - (
                (2.0 * num_col ** 2.0 + 3.0 * num_col - 1.0)
                / (6.0 * (num_col + 1.0) * (num_classes - 1.0))
            ) * (np.sum(1.0 / vec_weight) - 1.0 / (num_inst - num_classes))
            return gamma

        def calc_m_factor(
            sample_cov_matrices: np.ndarray,
            pooled_cov_mat: np.ndarray,
            num_inst: int,
            num_classes: int,
            gamma: float,
            vec_weight: np.ndarray,
        ) -> float:
            """Calculate the M factor."""
            vec_logdet = [
                np.log(np.linalg.det(S_i)) for S_i in sample_cov_matrices
            ]

            m_factor = gamma * (
                (num_inst - num_classes)
                * np.log(np.linalg.det(pooled_cov_mat))
                - np.dot(vec_weight, vec_logdet)
            )

            return m_factor

        num_inst, num_col = N.shape

        if classes is None or class_freqs is None:
            classes, class_freqs = np.unique(y, return_counts=True)

        _classes = np.asarray(classes)
        _class_freqs = np.asarray(class_freqs, dtype=int)

        num_classes = _classes.size
        sample_cov_matrices = calc_sample_cov_mat(N, y, ddof)
        vec_weight = _class_freqs - 1.0

        pooled_cov_mat = calc_pooled_cov_mat(
            sample_cov_matrices, vec_weight, num_inst, num_classes
        )

        gamma = calc_gamma_factor(num_col, num_classes, num_inst)

        m_factor = calc_m_factor(
            sample_cov_matrices,
            pooled_cov_mat,
            num_inst,
            num_classes,
            gamma,
            vec_weight,
        )

        if np.isinf(m_factor):
            return np.nan

        return float(np.exp(m_factor / (num_col * (num_inst - num_classes))))

    @classmethod
    def ft_skewness(
        cls, N: np.ndarray, method: int = 3, bias: bool = True
    ) -> np.ndarray:
        """Compute the skewness for each attribute.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        method : int, optional
            Defines the strategy used for estimate data skewness. This argument
            is used fo compatibility with R package `e1071`. The options must
            be one of the following:

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

            Where `n` is the number of instances in ``N``, `s` is the standard
            deviation of each attribute in ``N``, and `m_i` is the ith
            statistical momentum of each attribute in ``N``.

            Note that if the selected method is unable to be calculated due to
            division by zero, then the first method will be used instead.

        bias : bool, optional
            If False, then the calculations are corrected for statistical bias.

        Returns
        -------
        :obj:`np.ndarray`
            Attribute skewness.

        References
        ----------
        .. [1] Donald Michie, David J. Spiegelhalter, Charles C. Taylor, and
           John Campbell. Machine Learning, Neural and Statistical
           Classification, volume 37. Ellis Horwood Upper Saddle River, 1994.
        """
        skew_arr = np.apply_along_axis(
            func1d=_summary.sum_skewness,
            axis=0,
            arr=N,
            bias=bias,
            method=method,
        )

        return skew_arr

    @classmethod
    def ft_sparsity(cls, X: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Compute (possibly normalized) sparsity metric for each attribute.

        Sparsity `S` of a vector `v` of numeric values is defined as

            S(v) = (1.0 / (n - 1.0)) * ((n / phi(v)) - 1.0),

        where
            - `n` is the number of instances in dataset ``X``.
            - `phi(v)` is the number of distinct values in `v`.

        Parameters
        ----------
        X : :obj:`np.ndarray`
            Fitted numerical data.

        normalize : bool, optional
            If True, then the output will be S(v) as shown above. Otherwise,
            the output is not be multiplied by the `(1.0 / (n - 1.0))` factor
            (i.e. new output is defined as S'(v) = ((n / phi(v)) - 1.0)).

        Returns
        -------
        :obj:`np.ndarray`
            Attribute sparsities.

        References
        ----------
        .. [1] Mostafa A. Salama, Aboul Ella Hassanien, and Kenneth Revett.
           Employment of neural network and rough set in meta-learning.
           Memetic Computing, 5(3):165 – 177, 2013.
        """
        ans = np.array([attr.size / np.unique(attr).size for attr in X.T])

        num_inst, _ = X.shape

        norm_factor = 1.0
        if normalize:
            norm_factor = 1.0 / (num_inst - 1.0)

        return (ans - 1.0) * norm_factor

    @classmethod
    def ft_t_mean(cls, N: np.ndarray, pcut: float = 0.2) -> np.ndarray:
        """Compute the trimmed mean of each attribute.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        pcut : float, optional
            Percentage of cut from both the `lower` and `higher` values.
            This value should be in interval [0.0, 0.5), where if 0.0 the
            return value is the default mean calculation.

        Returns
        -------
        :obj:`np.ndarray`
            Attribute trimmed means.

        References
        ----------
        .. [1] Robert Engels and Christiane Theusinger. Using a data metric for
           preprocessing advice for data mining applications. In 13th European
           Conference on on Artificial Intelligence (ECAI), pages 430 – 434,
           1998.
        """
        return scipy.stats.trim_mean(N, proportiontocut=pcut)

    @classmethod
    def ft_var(cls, N: np.ndarray, ddof: int = 1) -> np.ndarray:
        """Compute the variance of each attribute.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        ddof : float, optional
            Degrees of freedom for variance.

        Returns
        -------
        :obj:`np.ndarray`
            Attribute variances.

        References
        ----------
        .. [1] Ciro Castiello, Giovanna Castellano, and Anna Maria Fanelli.
           Meta-data: Characterization of input features for meta-learning.
           In 2nd International Conference on Modeling Decisions for
           Artificial Intelligence (MDAI), pages 457–468, 2005.
        """
        return np.asfarray(N.var(axis=0, ddof=ddof))

    @classmethod
    def ft_w_lambda(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        can_cor_eigvals: t.Optional[np.ndarray] = None,
        can_cors: t.Optional[np.ndarray] = None,
    ) -> float:
        """Compute the Wilks' Lambda value.

        The Wilk's Lambda L is calculated as:

            L = prod(1.0 / (1.0 + can_cor_eig_i))

        Where `can_cor_eig_i` is the ith eigenvalue related to the ith
        canonical correlation `can_cor_i` between the attributes in ``N``
        and the binarized (one-hot encoded) version of ``y``.

        The relationship between `can_cor_eig_i` and `can_cor_i` is
        given by:

            can_cor_i = sqrt(can_cor_eig_i / (1 + can_cor_eig_i))

        Or, equivalently:

            can_cor_eig_i = can_cor_i**2 / (1 - can_cor_i**2)

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Fitted numerical data.

        y : :obj:`np.ndarray`
            Target attribute.

        can_cor_eigvals : :obj:`np.ndarray`, optional
            Eigenvalues associated with the canonical correlations of
            ``N`` and one-hot encoded ``y``. This argument is used to
            exploit precomputations. The relationship between the ith
            canonical correlation `can_cor_i` and its eigenvalue is:

                can_cor_i = sqrt(can_cor_eigval_i / (1 + can_cor_eigval_i))

            Or, equivalently:

                can_cor_eigval_i = can_cor_i**2 / (1 - can_cor_i**2)

        can_cors : :obj:`np.ndarray`, optional
            Canonical correlations between ``N`` and the one-hot encoded
            version of ``y``. Argument used to take advantage of
            precomputations. Used only if ``can_cor_eigvals`` is None.

        Returns
        -------
        float
            Wilk's lambda value.

        References
        ----------
        .. [1] Guido Lindner and Rudi Studer. AST: Support for algorithm
           selection with a CBR approach. In European Conference on
           Principles of Data Mining and Knowledge Discovery (PKDD),
           pages 418 – 423, 1999.
        """
        if can_cor_eigvals is None:
            if can_cors is None:
                can_cors = cls._calc_can_cors(N=N, y=y)

            can_cor_eigvals = cls._can_cor_to_eigval(can_cors)

        if can_cor_eigvals.size == 0:
            return np.nan

        # return float(np.prod(1 / (1 + can_cor_eigvals)))
        return float(np.exp(-np.sum(np.log1p(can_cor_eigvals))))

    @classmethod
    def ft_p_trace(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        can_cors: t.Optional[np.ndarray] = None,
    ) -> float:
        """Compute the Pillai's trace.

        The Pillai's trace is the sum of the squared canonical
        correlations of ``N`` and the one-hot encoded version of ``y``.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        can_cors : :obj:`np.ndarray`, optional
            Canonical correlations between ``N`` and the one-hot encoded
            version of ``y``. Argument used to take advantage of
            precomputations.

        Returns
        -------
        float
            Pillai's trace value.

        References
        ----------
        .. [1] Pillai K.C.S (1955). Some New test criteria in multivariate
           analysis. Ann Math Stat: 26(1):117–21. Seber, G.A.F. (1984).
           Multivariate Observations. New York: John Wiley and Sons.
        """
        if can_cors is None:
            can_cors = cls._calc_can_cors(N=N, y=y)

        if can_cors.size == 0:  # type: ignore
            return np.nan

        return float(np.sum(np.square(can_cors)))

    @classmethod
    def ft_lh_trace(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        can_cor_eigvals: t.Optional[np.ndarray] = None,
        can_cors: t.Optional[np.ndarray] = None,
    ) -> float:
        """Compute the Lawley-Hotelling trace.

        The Lawley-Hotelling trace LH is given by:

            LH = sum_{i} can_cor_i**2 / (1 - can_cor_i**2)

        Where `can_cor_i` is the ith canonical correlation of
        ``N`` and the one-hot encoded version of ``y``.

        Equivalently, LH can be calculated from the eigenvalues
        related to each canonical correlation due to the relationship:

            can_cor_eigval_i = can_cor_i**2 / (1 - can_cor_i**2)

        Therefore, LH is given simply by:

            LH = sum_{i} can_cor_eigval_i

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        can_cor_eigvals : :obj:`np.ndarray`, optional
            Eigenvalues associated with the canonical correlations of
            ``N`` and one-hot encoded ``y``. This argument is used to
            exploit precomputations. The relationship between the ith
            canonical correlation ``can_cor_i`` and its eigenvalue is:

                can_cor_i = sqrt(can_cor_eigval_i / (1 + can_cor_eigval_i))

            Or, equivalently:

                can_cor_eigval_i = can_cor_i**2 / (1 - can_cor_i**2)

        can_cors : :obj:`np.ndarray`, optional
            Canonical correlations between ``N`` and the one-hot encoded
            version of ``y``. Argument used to take advantage of
            precomputations. Used only if ``can_cor_eigvals`` is None.

        Returns
        -------
        float
            Lawley-Hotelling trace value.

        References
        ----------
        .. [1] Lawley D. A Generalization of Fisher’s z Test. Biometrika.
           1938;30(1):180-187.
        .. [2] Hotelling H. A generalized T test and measure of multivariate
           dispersion. In: Neyman J, ed. Proceedings of the Second Berkeley
           Symposium on Mathematical Statistics and Probability. Berkeley:
           University of California Press; 1951:23-41.
        """
        if can_cor_eigvals is None:
            if can_cors is None:
                can_cors = cls._calc_can_cors(N=N, y=y)

            can_cor_eigvals = cls._can_cor_to_eigval(can_cors)

        if can_cor_eigvals.size == 0:  # type: ignore
            return np.nan

        return float(np.sum(can_cor_eigvals))

    @classmethod
    def ft_roy_root(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        criterion: str = "eigval",
        can_cors: t.Optional[np.ndarray] = None,
        can_cor_eigvals: t.Optional[np.ndarray] = None,
    ) -> float:
        """Compute the Roy's largest root.

        The Roy's largest root RLR can be computed using two distinct
        approaches (see references for further explanation.)

        1. Based on Roy's (ii) original hypothesis: formulated using the
        largest eigenvalue associated with the canonical correlations
        between ``N`` and the one-hot encoded version of ``y``. That
        is, the Roy's Largest Root RLR_a can be defined as:

            RLR_a = max_{I} can_cor_eig_val_i

        It is in range [0, +inf).

        2. Based on Roy's (iii) original hypothesis: formulated using
        the largest squared canonical correlations of ``N``and the one-
        hot encoded version of ``y``. Therefore, the Roy's Largest Root
        RLR_b can be defined as:

            RLR_b = max_{i} can_cor_i**2

        It is in range [0, 1].

        Note that both statistics have different meanings and, therefore,
        will assume distinct values.

        Which formulation is used can be controled using the ``criterion``
        argument (see below for more information.)

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        criterion : str, optional
            If `eigval`, calculate the Roy's largest root as the largest
            eigenvalue associated with each canonical correlation.
            This is the first formulation described above. If `cancor`,
            calculate the Roy's largest root as the largest squared
            canonical correlation. This is the second formulation above.

        can_cors : :obj:`np.ndarray`, optional
            Canonical correlations between ``N`` and the one-hot encoded
            version of ``y``. Argument used to take advantage of
            precomputations. Used only if ``criterion`` is `cancor` or,
            if otherwise, ``can_cor_eigvals`` argument is None.

        can_cor_eigvals : :obj:`np.ndarray`, optional
            Eigenvalues associated with the canonical correlations of
            ``N`` and one-hot encoded ``y``. This argument is used to
            exploit precomputations. The relationship between the ith
            canonical correlation ``can_cor_i`` and its eigenvalue is:

                can_cor_i = sqrt(can_cor_eigval_i / (1 + can_cor_eigval_i))

            Or, equivalently:

                can_cor_eigval_i = can_cor_i**2 / (1 - can_cor_i**2)

            This argument is used only if ``criterion`` argument is
            `eigval`.

        Returns
        -------
        float
            Roy's largest root calculated based on criterion defined by the
            ``criterion`` argument.

        References
        ----------
        .. [1] Roy SN. On a Heuristic Method of Test Construction and its
           use in Multivariate Analysis. Ann Math Stat. 1953;24(2):220-238.
        .. [2] A note on Roy's largest root. Kuhfeld, W.F. Psychometrika (1986)
           51: 479. https://doi.org/10.1007/BF02294069
        """
        VALID_CRITERIA = ("eigval", "cancor")

        if criterion not in VALID_CRITERIA:
            raise ValueError(
                "Roy's largest root 'criterion' must be in {}.".format(
                    VALID_CRITERIA
                )
            )

        if criterion == "eigval":
            if can_cor_eigvals is None:
                if can_cors is None:
                    can_cors = cls._calc_can_cors(N=N, y=y)

                can_cor_eigvals = cls._can_cor_to_eigval(can_cors)

            values = can_cor_eigvals

        else:
            if can_cors is None:
                can_cors = cls._calc_can_cors(N=N, y=y)

            values = np.square(can_cors)

        if values.size == 0:  # type: ignore
            return np.nan

        return float(np.max(values))
