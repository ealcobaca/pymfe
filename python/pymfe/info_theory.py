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
import sklearn.metrics


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
    def _entropy(cls, values: t.Union[np.ndarray, t.List]) -> float:
        """Calculate Shannon entropy within array ``values``.

        Check ``ft_attr_ent`` and ``ft_class_end`` methods for more
        information.
        """
        _, counts = np.unique(values, return_counts=True)
        return scipy.stats.entropy(counts, base=2)

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
    def _conc(cls,
              vec_x: np.ndarray,
              vec_y: np.ndarray,
              epsilon: float = 1.0e-8) -> float:
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
    def ft_attr_ent(cls, C: np.ndarray) -> t.Union[np.ndarray, np.float]:
        """Calculates Shannon entropy for each predictive attribute.

        The Shannon Entropy H of a vector x is defined as:

            H(x) = - sum_{val in phi_x}(P(x = val) * log2(P(x = val))

        Where phi_x is a set of all possible distinct values in
        vector x and P(x = val) is the probability of x assume some
        value val in phi_x.
        """
        try:
            return np.apply_along_axis(
                func1d=MFEInfoTheory._entropy, axis=0, arr=C)

        except ValueError:
            return np.nan

    @classmethod
    def ft_class_conc(cls, C: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute concentration coefficient between each attr. and class."""
        return np.apply_along_axis(
            func1d=MFEInfoTheory._conc, axis=0, arr=C, vec_y=y)

    @classmethod
    def ft_class_ent(cls, y: np.ndarray) -> float:
        """Calculates target attribute Shannon entropy.

        The Shannon Entropy H of a vector y is defined as:

            H(y) = - sum_{val in phi_y}(P(y = val) * log2(P(y = val))

        Where phi_y is a set of all possible distinct values in
        vector y and P(y = val) is the probability of y assume some
        value val in phi_y.
        """
        return MFEInfoTheory._entropy(y)

    @classmethod
    def ft_eq_num_attr(cls,
                       C: np.ndarray,
                       y: np.ndarray,
                       epsilon: float = 1.0e-8) -> float:
        """Number of attributes equivalent for a predictive task.

        The attribute equivalence E is defined as:

            E = attr_num * (H(y) / sum_x(MI(x, y)))

        Where H(y) is the Shannon Entropy of the target attribute and
        MI(x, y) is the Mutual Information between the predictive att-
        ribute x and target attribute y.

        Args:
            epsilon (:obj:`float`, optional): small numeric value to
                avoid division by zero.
        """
        ent_class = MFEInfoTheory._entropy(y)
        mutual_info = MFEInfoTheory.ft_mut_inf(C, y)

        _, num_col = C.shape

        return num_col * (ent_class / (epsilon + sum(mutual_info)))

    @classmethod
    def ft_joint_ent(cls, C: np.ndarray, y: np.ndarray) -> np.ndarray:
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

        def joint_entropy(vec_x: np.ndarray,
                          y: np.ndarray,
                          epsilon: float = 1.0e-8) -> float:
            joint_prob_mat = MFEInfoTheory._joint_prob_mat(vec_x,
                                                           y) + epsilon

            joint_entropy = np.multiply(joint_prob_mat,
                                        np.log2(joint_prob_mat)).sum().sum()

            return -1.0 * joint_entropy

        joint_entropy_array = np.apply_along_axis(
            func1d=joint_entropy, axis=0, arr=C, y=y)

        return joint_entropy_array

    @classmethod
    def ft_mut_inf(cls, C: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Mutual information between each attribute in C and class y.

        Mutual Information MI between an attribute x and target attri-
        bute y is defined as:

            MI(x, y) = H(x) + H(y) - H(x, y)

        Where H(x) and H(y) are, respectively, the Shannon Entropy (see
        ``ft_attr_ent`` or ``ft_class_ent`` documentations for more in-
        formation) for x and y and H(x, y) is the Joint Entropy between
        x and y (see ``ft_joint_ent`` documentation for more informa-
        tion).
        """
        return np.apply_along_axis(
            func1d=sklearn.metrics.mutual_info_score,
            axis=0,
            arr=C,
            labels_pred=y)

    @classmethod
    def ft_ns_ratio(cls,
                    C: np.ndarray,
                    y: np.ndarray,
                    epsilon: float = 1.0e-8) -> float:
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
        ent_attr = sum(MFEInfoTheory.ft_attr_ent(C))
        mutual_info = sum(MFEInfoTheory.ft_mut_inf(C, y))

        return (ent_attr - mutual_info) / (epsilon + mutual_info)
