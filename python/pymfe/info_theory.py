"""Module dedicated to extraction of Information Theory Metafeatures.


Notes:
    For more information about the metafeatures implemented here,
    check out `Rivolli et al.`_.

References:
    .. _Rivolli et al.:
        "Towards Reproducible Empirical Research in Meta-Learning",
        Rivolli et al. URL: https://arxiv.org/abs/1808.10406
"""
import typing as t
import itertools

import numpy as np
import scipy


class MFEInfoTheory:
    """Keeps methods for metafeatures of ``Information Theory`` group.

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
    def precompute_class_freq(cls, y: t.Optional[np.ndarray] = None,
                              **kwargs) -> t.Dict[str, t.Any]:
        """Precompute classe (absolute) frequencies.

        Args:
            y (:obj:`np.ndarray`, optional): target attribute vector.

            **kwargs: extra arguments. May containg values thar are already
                precomputed before this method, so it can help speed up the
                precomputation.

        Return:
            dict: with following precomputed items:

                - ``class_freqs`` (:obj:`np.ndarray`): absolute frequency of
                    each distinct class in ``y``, if ``y`` is not :obj:`None-
                    Type`.
        """
        precomp_vals = {}

        if y is not None and "class_freqs" not in kwargs:
            _, class_freqs = np.unique(y, return_counts=True)

            precomp_vals["class_freqs"] = class_freqs

        return precomp_vals

    @classmethod
    def precompute_entropy(cls,
                           y: t.Optional[np.ndarray] = None,
                           C: t.Optional[np.ndarray] = None,
                           **kwargs) -> t.Dict[str, t.Any]:
        """Precompute various values related to Shannon's Entropy.

        Args:
            y (:obj:`np.ndarray`, optional): target attribute vector.

            C (:obj:`np.ndarray`, optional): categorical attributes from fitted
                data.

            **kwargs: extra arguments. May containg values thar are already
                precomputed before this method, so it can help speed up the
                precomputation.

        Return:
            dict: with following precomputed items:

                - ``class_ent`` (:obj:`np.ndarray`): Shannon's Entropy of ``y``
                    if it is not :obj:`NoneType`.

                - ``attr_ent`` (:obj:`np.ndarray`): Shannon's Entropy of each
                    attribute in ``C``, if it is not :obj:`NoneType`.

                - ``joint_ent`` (:obj:`np.ndarray`): Joint Entropy between each
                    attribute in ``C`` and target attribute ``y`` if both are
                    not :obj:`NoneType`.

                - ``mut_inf`` (:obj:`np.ndarray`): mutual information between
                    each attribute in ``C`` and ``y``, if they both are not
                    :obj:`NoneType`.
        """
        precomp_vals = {}

        if y is not None and "class_ent" not in kwargs:
            precomp_vals["class_ent"] = MFEInfoTheory.ft_class_ent(y)

        if C is not None and "attr_ent" not in kwargs:
            precomp_vals["attr_ent"] = MFEInfoTheory.ft_attr_ent(C)

        if y is not None and C is not None:
            if "joint_ent" not in kwargs:
                precomp_vals["joint_ent"] = np.apply_along_axis(
                    func1d=MFEInfoTheory._joint_ent, axis=0, arr=C, vec_y=y)

            if "mut_inf" not in kwargs:
                precomp_vals["mut_inf"] = MFEInfoTheory.ft_mut_inf(
                    C=C, y=y,
                    attr_ent=precomp_vals.get("attr_ent"),
                    class_ent=precomp_vals.get("class_ent"),
                    joint_ent=precomp_vals.get("joint_ent"))

        return precomp_vals

    @classmethod
    def _entropy(cls,
                 values: t.Union[np.ndarray, t.List],
                 class_freqs: t.Optional[np.ndarray] = None) -> float:
        """Calculate Shannon's entropy within array ``values``.

        Check ``ft_attr_ent`` and ``ft_class_end`` methods for more
        information.
        """
        if class_freqs is None:
            _, class_freqs = np.unique(values, return_counts=True)

        return scipy.stats.entropy(class_freqs, base=2)

    @classmethod
    def _joint_prob_mat(cls, vec_x: np.ndarray,
                        vec_y: np.ndarray) -> np.ndarray:
        """Compute joint probability matrix P(a, b), a in vec_x and b in vec_y.

        Used for ``_conc`` method and ``ft_joint_ent``.
        """
        x_vals = np.unique(vec_x)
        y_vals = np.unique(vec_y)

        joint_prob_mat = np.array([
            sum((vec_x == x_val) & (vec_y == y_val))
            for y_val, x_val in itertools.product(y_vals, x_vals)
        ]).reshape((y_vals.size, x_vals.size)) / vec_x.size

        return joint_prob_mat

    @classmethod
    def _joint_ent(cls,
                   vec_x: np.ndarray,
                   vec_y: np.ndarray,
                   epsilon: float = 1.0e-10) -> float:
        """Compute joint entropy between vectorx ``x`` and ``y``."""
        joint_prob_mat = MFEInfoTheory._joint_prob_mat(vec_x, vec_y) + epsilon

        joint_ent = np.multiply(joint_prob_mat,
                                np.log2(joint_prob_mat)).sum().sum()

        return -1.0 * joint_ent

    @classmethod
    def _conc(cls,
              vec_x: np.ndarray,
              vec_y: np.ndarray,
              epsilon: float = 1.0e-10) -> float:
        """Concentration coefficient between two arrays ``vec_x`` and ``vec_y``.

        Used for methods ``ft_class_conc`` and ``ft_attr_conc``.

        Args:
            epsilon (:obj:`float`, optional): small numeric value to
                avoid division by zero.
        """
        pij = MFEInfoTheory._joint_prob_mat(vec_x, vec_y)

        isum = pij.sum(axis=1)
        jsum2 = sum(pij.sum(axis=0)**2.0)

        conc = ((((pij**2.0).T / isum).sum().sum() - jsum2) /
                (1.0 - jsum2 + epsilon))

        return conc

    @classmethod
    def ft_attr_conc(cls, C: np.ndarray) -> np.ndarray:
        """Compute concentration coef. of each pair of distinct attributes."""

        _, num_col = C.shape

        col_permutations = itertools.permutations(range(num_col), 2)

        attr_conc = np.array([
            MFEInfoTheory._conc(C[:, col_a], C[:, col_b])
            for col_a, col_b in col_permutations
        ])

        return attr_conc

    @classmethod
    def ft_attr_ent(cls,
                    C: np.ndarray,
                    attr_ent: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Calculates Shannon's entropy for each predictive attribute.

        The Shannon's Entropy H of a vector x is defined as:

            H(x) = - sum_{val in phi_x}(P(x = val) * log2(P(x = val))

        Where phi_x is a set of all possible distinct values in
        vector x and P(x = val) is the probability of x assume some
        value val in phi_x.
        """
        if attr_ent is not None:
            return attr_ent

        try:
            return np.apply_along_axis(
                func1d=MFEInfoTheory._entropy, axis=0, arr=C)

        except ValueError:
            return np.array([np.nan])

    @classmethod
    def ft_class_conc(cls, C: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute concentration coefficient between each attr. and class."""
        return np.apply_along_axis(
            func1d=MFEInfoTheory._conc, axis=0, arr=C, vec_y=y)

    @classmethod
    def ft_class_ent(cls,
                     y: np.ndarray,
                     class_ent: t.Optional[np.ndarray] = None) -> float:
        """Calculates target attribute Shannon's entropy.

        The Shannon's Entropy H of a vector y is defined as:

            H(y) = - sum_{val in phi_y}(P(y = val) * log2(P(y = val))

        Where phi_y is a set of all possible distinct values in
        vector y and P(y = val) is the probability of y assume some
        value val in phi_y.
        """
        if class_ent is not None:
            return class_ent

        return MFEInfoTheory._entropy(y)

    @classmethod
    def ft_eq_num_attr(cls,
                       C: np.ndarray,
                       y: np.ndarray,
                       epsilon: float = 1.0e-10,
                       class_ent: t.Optional[np.ndarray] = None,
                       mut_inf: t.Optional[np.ndarray] = None) -> float:
        """Number of attributes equivalent for a predictive task.

        The attribute equivalence E is defined as:

            E = attr_num * (H(y) / sum_x(MI(x, y)))

        Where H(y) is the Shannon's Entropy of the target attribute and
        MI(x, y) is the Mutual Information between the predictive att-
        ribute x and target attribute y.

        Args:
            epsilon (:obj:`float`, optional): small numeric value to
                avoid division by zero.
        """
        if class_ent is None:
            class_ent = MFEInfoTheory._entropy(y)

        if mut_inf is None:
            mut_inf = MFEInfoTheory.ft_mut_inf(C, y)

        _, num_col = C.shape

        return num_col * (class_ent / (epsilon + sum(mut_inf)))

    @classmethod
    def ft_joint_ent(cls,
                     C: np.ndarray,
                     y: np.ndarray,
                     joint_ent: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate Joint entropy between each attribute and class.

        The Joint Entropy H between a predictive attribute x and target
        attribute y is defined as:

            H(x, y) = - sum_{phi_x}(sum_{phi_y}(p_i_j * log2(p_i_j)))

        Where phi_x and phi_y are sets of possible distinct values for,
        respectively, x and y and p_i_j is defined as:

            p_i_j = P(x = phi_x_i, y = phi_y_j)

        That is, p_i_j is the joint probability of x to assume a specific
        value i in the set phi_x simultaneously with y assuming a specific
        value j in the set phi_y.
        """
        if joint_ent is None:
            joint_ent = np.apply_along_axis(
                func1d=MFEInfoTheory._joint_ent, axis=0, arr=C, vec_y=y)

        return joint_ent

    @classmethod
    def ft_mut_inf(cls,
                   C: np.ndarray,
                   y: np.ndarray,
                   mut_inf: t.Optional[np.ndarray] = None,
                   attr_ent: t.Optional[np.ndarray] = None,
                   class_ent: t.Optional[np.ndarray] = None,
                   joint_ent: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Mutual information between each attribute in C and class y.

        Mutual Information MI between an attribute x and target attri-
        bute y is defined as:

            MI(x, y) = H(x) + H(y) - H(x, y)

        Where H(x) and H(y) are, respectively, the Shannon's Entropy (see
        ``ft_attr_ent`` or ``ft_class_ent`` documentations for more in-
        formation) for x and y and H(x, y) is the Joint Entropy between
        x and y (see ``ft_joint_ent`` documentation for more informa-
        tion).
        """
        if mut_inf is not None:
            return mut_inf

        if class_ent is None:
            class_ent = MFEInfoTheory._entropy(y)

        if attr_ent is None:
            attr_ent = np.apply_along_axis(
                func1d=MFEInfoTheory.ft_attr_ent, axis=0, arr=C)

        if joint_ent is None:
            joint_ent = np.apply_along_axis(
                func1d=MFEInfoTheory._joint_ent, axis=0, arr=C, vec_y=y)

        return attr_ent + class_ent - joint_ent

    @classmethod
    def ft_ns_ratio(cls,
                    C: np.ndarray,
                    y: np.ndarray,
                    epsilon: float = 1.0e-10,
                    attr_ent: t.Optional[np.ndarray] = None,
                    mut_inf: t.Optional[np.ndarray] = None) -> float:
        """Compute noisiness of attributes.

        Let y be a target attribute and x one predictive attribute in
        a dataset D. Noisiness N is defined as:

            N = (sum_x(attr_entropy(x)) - sum_x(MI(x, y))) / sum_x(MI(x, y))

        where MI(x, y) is the Mutual Information between class attri-
        bute and predictive attribute x, and all ``sum`` is over all
        distinct predictive attributes in D.

        Args:
            epsilon (:obj:`float`, optional): small numeric value to
                avoid division by zero.
        """
        if attr_ent is None:
            attr_ent = MFEInfoTheory.ft_attr_ent(C)

        if mut_inf is None:
            mut_inf = MFEInfoTheory.ft_mut_inf(C, y)

        ent_attr = sum(attr_ent)
        mut_inf = sum(mut_inf)

        return (ent_attr - mut_inf) / (epsilon + mut_inf)
