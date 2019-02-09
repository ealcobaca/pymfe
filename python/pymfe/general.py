"""A module dedicated to the extraction of General Metafeatures.

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


class MFEGeneral:
    """Keep methods for metafeatures of ``General``/``Simple`` group.

    The convention adopted for metafeature extraction related methods is to
    always start with ``ft_`` prefix to allow automatic method detection. This
    prefix is predefined within ``_internal`` module.

    All method signature follows the conventions and restrictions listed below:
        1. For independent attribute data, ``X`` means ``every type of attribu-
            te``, ``N`` means ``Numeric attributes only`` and ``C`` stands for
            ``Categorical attributes only``.

        2. Only ``X``, ``y``, ``N``, ``C`` and ``splits`` are allowed to be re-
            quired method arguments. All other arguments must be strictly opti-
            onal (i.e., has a predefined default value).

        3. The initial assumption is that the user can change any optional ar-
            gument, without any previous verification of argument value or its
            type, via **kwargs argument of ``extract`` method of MFE class.

        4. The return value of all feature extraction methods should be a sin-
            gle value or a generic Sequence (preferably a :obj:`np.ndarray`)
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
    def precompute_general_class(cls,
                                 y: t.Optional[np.ndarray] = None,
                                 **kwargs) -> t.Dict[str, t.Any]:
        """Precompute distinct classes and its frequencies from ``y``.

        Args:
            y (:obj:`np.ndarray`, optional): target attribute from fitted data.

            **kwargs: additional arguments. May have previously precomputed be-
                fore this method from other precomputed methods, so they can
                help speed up this precomputation.

        Return:
            dict: with following precomputed items:
                - ``classes`` (:obj:`np.ndarray`): distinct classes of ``y``,
                    if ``y`` is not :obj:`NoneType`.
                - ``class_freqs`` (:obj:`np.ndarray`): class frequencies of
                    ``y``, if ``y`` is not :obj:`NoneType`.
        """
        precomp_vals = {}

        if y is not None and not {"classes", "class_freqs"}.issubset(kwargs):
            classes, class_freqs = np.unique(y, return_counts=True)

            precomp_vals["classes"] = classes
            precomp_vals["class_freqs"] = class_freqs

        return precomp_vals

    @classmethod
    def ft_attr_to_inst(cls, X: np.ndarray) -> int:
        """Returns ration between the number of attributes and instances.

        It is effectively the inverse of value given by ``ft_inst_to_attr``.
        """
        return X.shape[1] / X.shape[0]

    @classmethod
    def ft_cat_to_num(cls, C: np.ndarray,
                      N: np.ndarray) -> t.Union[int, np.float]:
        """Returns ratio between the number of categoric and numeric features.

        If the number of numeric features is zero, :obj:`np.nan` is returned
        instead.

        Effectively the inverse of value given by ``ft_num_to_cat``.
        """
        if N.shape[1] == 0:
            return np.nan

        return C.shape[1] / N.shape[1]

    @classmethod
    def ft_freq_class(cls, y: np.ndarray, class_freqs: np.ndarray = None
                      ) -> t.Union[np.ndarray, np.float]:
        """Returns an array of the relative frequency of each distinct class.

        Args:
            class_freqs (:obj:`np.ndarray`, optional): vector of (absolute,
                not relative) frequency of each class in data.
        """
        if y.size == 0:
            return np.nan

        if class_freqs is None:
            _, class_freqs = np.unique(y, return_counts=True)

        return class_freqs / y.size

    @classmethod
    def ft_inst_to_attr(cls, X: np.ndarray) -> int:
        """Returns the ratio between the number of instances and attributes.

        It is effectively the inverse of value given by ``ft_attr_to_inst``.
        """
        return X.shape[0] / X.shape[1]

    @classmethod
    def ft_nr_attr(cls, X: np.ndarray) -> int:
        """Returns the number of total attributes."""
        return X.shape[1]

    @classmethod
    def ft_nr_bin(cls, X: np.ndarray) -> int:
        """Returns the number of binary attributes."""
        bin_cols = np.apply_along_axis(
            func1d=lambda col: np.unique(col).size == 2, axis=0, arr=X)

        return sum(bin_cols)

    @classmethod
    def ft_nr_cat(cls, C: np.ndarray) -> int:
        """Returns the number of categorical attributes."""
        return C.shape[1]

    @classmethod
    def ft_nr_class(
            cls,
            y: t.Optional[np.ndarray] = None,
            classes: t.Optional[np.ndarray] = None) -> t.Union[float, int]:
        """Returns the number of distinct classes.

        ``y`` and ``classes`` can not be :obj:`NoneType` simultaneously,
        or else :obj:`np.nan` will be returned.

        Args:
            y (:obj:`np.ndarray`, optional): target vector.

            classes (:obj:`np.ndarray`, optional): vector with all distinct
                classes. This argument purpose is mainly for benefit from
                precomputations.

        Return:
            int or float: number of distinct classes in a target vector if
                either ``y`` or ``classes`` is given. Otherwise, returns
                :obj:`np.nan`.
        """
        if classes is not None:
            return classes.size

        if y is None:
            return np.nan

        return np.unique(y).size

    @classmethod
    def ft_nr_inst(cls, X: np.ndarray) -> int:
        """Returns the number of instances (rows) in the dataset."""
        return X.shape[0]

    @classmethod
    def ft_nr_num(cls, N: np.ndarray) -> int:
        """Returns the number of numeric features."""
        return N.shape[1]

    @classmethod
    def ft_num_to_cat(cls, C: np.ndarray,
                      N: np.ndarray) -> t.Union[int, np.float]:
        """Returns the ratio between the number of numeric and categoric features.

        If the number of categoric features is zero, :obj:`np.nan` is returned
        instead.

        Effectively the inverse of the value given by ``ft_cat_to_num``.
        """
        if C.shape[1] == 0:
            return np.nan

        return N.shape[1] / C.shape[1]
