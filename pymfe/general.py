"""A module dedicated to the extraction of general metafeatures.
"""
import typing as t
import numpy as np


class MFEGeneral:
    """Keep methods for metafeatures of ``General``/``Simple`` group.

    The convention adopted for metafeature extraction related methods is to
    always start with ``ft_`` prefix to allow automatic method detection. This
    prefix is predefined within ``_internal`` module.

    All method signature follows the conventions and restrictions listed below:

    1. For independent attribute data, ``X`` means ``every type of
       attribute``, ``N`` means ``Numeric attributes only`` and ``C`` stands
       for ``Categorical attributes only``. It is important to note that the
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
       value or a generic List (preferably a :obj:`np.ndarray`)
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
    def precompute_general_class(
        cls, y: t.Optional[np.ndarray] = None, **kwargs
    ) -> t.Dict[str, t.Any]:
        """Precompute distinct classes and its frequencies from ``y``.

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Target attribute.

        **kwargs
            Additional arguments. May have previously precomputed before
            this method from other precomputed methods, so they can help
            speed up this precomputation.

        Returns
        -------
        :obj:`dict`
            The following precomputed items are returned:
                * ``classes`` (:obj:`np.ndarray`):  distinct classes of
                  ``y``, if ``y`` is not :obj:`NoneType`.
                * ``class_freqs`` (:obj:`np.ndarray`): class frequencies of
                  ``y``, if ``y`` is not :obj:`NoneType`.
        """
        precomp_vals = {}

        if y is not None and not {"classes", "class_freqs"}.issubset(kwargs):
            classes, class_freqs = np.unique(y, return_counts=True)

            precomp_vals["classes"] = classes
            precomp_vals["class_freqs"] = class_freqs

        return precomp_vals

    @classmethod
    def ft_attr_to_inst(cls, X: np.ndarray) -> float:
        """Compute the ratio between the number of attributes.

        It is effectively the inverse of value given by ``ft_inst_to_attr``.

        Parameters
        ----------
        X : :obj:`np.ndarray`
            Fitted data.

        Returns
        -------
        float
            The ratio between the number of attributes and instances.

        References
        ----------
        .. [1] Alexandros Kalousis and Theoharis Theoharis. NOEMON: Design,
           implementation and performance results of an intelligent assistant
           for classifier selection. Intelligent Data Analysis, 3(5):319–337,
           1999.
        """
        return X.shape[1] / X.shape[0]

    @classmethod
    def ft_cat_to_num(
        cls, X: np.ndarray, cat_cols: t.List[int]
    ) -> t.Union[int, float]:
        """Compute the ratio between the number of categoric and numeric
        features.

        If the number of numeric features is zero, :obj:`np.nan` is returned
        instead.

        Effectively the inverse of value given by ``ft_num_to_cat``.

        Parameters
        ----------
        X : :obj:`np.ndarray`
            Fitted data.

        cat_cols : :obj:`list` of int
            List containing the indices of each categorical column
            in ``X``.

        Returns
        -------
        int or float
            Proportion of categorical and numerical attributes.

        References
        ----------
        .. [1] Matthias Feurer, Jost Tobias Springenberg, and Frank Hutter.
           Using meta-learning toinitialize bayesian optimization of
           hyperparameters. In International Conference on Meta-learning and
           Algorithm Selection (MLAS), pages 3 – 10, 2014.
        """
        num_cat = len(cat_cols)

        if X.shape[1] == num_cat:
            return np.nan

        return num_cat / (X.shape[1] - num_cat)

    @classmethod
    def ft_freq_class(
        cls,
        y: np.ndarray,
        class_freqs: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the relative frequency of each distinct class.

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Target attribute.

        class_freqs : :obj:`np.ndarray`, optional
            Absolute frequency of each distinct class. Argument
            used to take advantage of precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            Relative frequency of each distinct class.

        References
        ----------
        .. [1] Guido Lindner and Rudi Studer. AST: Support for algorithm
           selection with a CBR approach. In European Conference on
           Principles of Data Mining and Knowledge Discovery (PKDD),
           pages 418 – 423, 1999.
        """
        if class_freqs is None:
            _, class_freqs = np.unique(y, return_counts=True)

        return class_freqs / y.size

    @classmethod
    def ft_inst_to_attr(cls, X: np.ndarray) -> float:
        """Compute the ratio between the number of instances and attributes.

        It is effectively the inverse of value given by ``ft_attr_to_inst``.

        Parameters
        ----------
        X : :obj:`np.ndarray`
            Fitted data.

        Returns
        -------
        float
            Ratio of number of instances and number of predictive attributes.

        References
        ----------
        .. [1] Petr Kuba, Pavel Brazdil, Carlos Soares, and Adam Woznica.
           Exploiting sampling andmeta-learning for parameter setting for
           support vector machines. In 8th IBERAMIA Workshop on Learning
           and Data Mining, pages 209 – 216, 2002.
        """
        return X.shape[0] / X.shape[1]

    @classmethod
    def ft_nr_attr(cls, X: np.ndarray) -> int:
        """Compute the total number of attributes.

        Parameters
        ----------
        X : :obj:`np.ndarray`
            Fitted data.

        Returns
        -------
        int
            Total number of attributes in the data without transformations.

        References
        ----------
        .. [1] Donald Michie, David J. Spiegelhalter, Charles C. Taylor, and
           John Campbell. Machine Learning, Neural and Statistical
           Classification, volume 37. Ellis Horwood Upper Saddle River, 1994.
        """
        return X.shape[1]

    @classmethod
    def ft_nr_bin(cls, X: np.ndarray) -> int:
        """Compute the number of binary attributes.

        Any attribute that has exactly two distinct values is considered
        a binary attribute, independently of its data type.

        Parameters
        ----------
        X : :obj:`np.ndarray`
            Fitted data.

        Returns
        -------
        int
            Number of binary attributes in ``X``.

        References
        ----------
        .. [1] Donald Michie, David J. Spiegelhalter, Charles C. Taylor, and
           John Campbell. Machine Learning, Neural and Statistical
           Classification, volume 37. Ellis Horwood Upper Saddle River, 1994.
        """
        bin_cols = np.apply_along_axis(
            func1d=lambda col: np.unique(col).size == 2, axis=0, arr=X
        )

        return np.sum(bin_cols)

    @classmethod
    def ft_nr_cat(cls, cat_cols: t.List[int]) -> int:
        """Compute the number of categorical attributes.

        Parameters
        ----------
        cat_cols : :obj:`list` of int
            List containing the indices of each categorical column
            in ``X``.

        Returns
        -------
        int
            Number of categorical attributes in ``X``.

        References
        ----------
        .. [1] Robert Engels and Christiane Theusinger. Using a data metric for
           preprocessing advice for data mining applications. In 13th European
           Conference on on Artificial Intelligence (ECAI), pages 430 – 434,
           1998.
        """
        return len(cat_cols)

    @classmethod
    def ft_nr_class(
        cls, y: np.ndarray, classes: t.Optional[np.ndarray] = None
    ) -> int:
        """Compute the number of distinct classes.

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Target attribute.

        classes : :obj:`np.ndarray`, optional
            Array with all distinct classes. This argument purpose is
            mainly for benefit from precomputations.

        Returns
        -------
        int
            Number of distinct classes in ``y``.

        References
        ----------
        .. [1] Donald Michie, David J. Spiegelhalter, Charles C. Taylor, and
           John Campbell. Machine Learning, Neural and Statistical
           Classification, volume 37. Ellis Horwood Upper Saddle River, 1994.
        """
        if classes is not None:
            return classes.size

        return np.unique(y).size

    @classmethod
    def ft_nr_inst(cls, X: np.ndarray) -> int:
        """Compute the number of instances (rows) in the dataset.

        Parameters
        ----------
        X : :obj:`np.ndarray`
            Fitted data.

        Returns
        -------
        int
            Number of instances in ``X``.

        References
        ----------
        .. [1] Donald Michie, David J. Spiegelhalter, Charles C. Taylor, and
           John Campbell. Machine Learning, Neural and Statistical
           Classification, volume 37. Ellis Horwood Upper Saddle River, 1994.
        """
        return X.shape[0]

    @classmethod
    def ft_nr_num(cls, X: np.ndarray, cat_cols: t.List[int]) -> int:
        """Compute the number of numeric features.

        Parameters
        ----------
        X : :obj:`np.ndarray`
            Fitted data.

        cat_cols : :obj:`list` of int
            List containing the indices of each categorical column
            in ``X``.

        Returns
        -------
        int
            Number of numerical attributes in ``X``.

        References
        ----------
        .. [1] Robert Engels and Christiane Theusinger. Using a data metric for
           preprocessing advice for data mining applications. In 13th European
           Conference on on Artificial Intelligence (ECAI), pages 430 – 434,
           1998.
        """
        return X.shape[1] - len(cat_cols)

    @classmethod
    def ft_num_to_cat(
        cls, X: np.ndarray, cat_cols: t.List[int]
    ) -> t.Union[int, float]:
        """Compute the number of numerical and categorical features.

        If the number of categoric features is zero, :obj:`np.nan` is returned
        instead.

        Effectively the inverse of the value given by ``ft_cat_to_num``.

        Parameters
        ----------
        X : :obj:`np.ndarray`
            Fitted data.

        cat_cols : :obj:`list` of int
            List containing the indices of each categorical column
            in ``X``.

        Returns
        -------
        int or float
            If ``X`` has at least one categorical feature, then return the
            ratio of numerical and categorical features. Return :obj:`np.nan`
            otherwise.

        References
        ----------
        .. [1] Matthias Feurer, Jost Tobias Springenberg, and Frank Hutter.
           Using meta-learning toinitialize bayesian optimization of
           hyperparameters. In International Conference on Meta-learning and
           Algorithm Selection (MLAS), pages 3 – 10, 2014.
        """
        if not cat_cols:
            return np.nan

        num_cat = len(cat_cols)

        return (X.shape[1] - num_cat) / num_cat
