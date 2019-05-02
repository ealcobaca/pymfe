"""A module dedicated to the extraction of Information Theoretic Metafeatures.
"""
import typing as t
import itertools

import pandas as pd
import numpy as np
import scipy


class MFEInfoTheory:
    """Keeps methods for metafeatures of ``Information Theory`` group.

    The convention adopted for metafeature extraction related methods is to
    always start with ``ft_`` prefix to allow automatic method detection. This
    prefix is predefined within ``_internal`` module.

    All method signature follows the conventions and restrictions listed below:

    1. For independent attribute data, ``X`` means ``every type of attribute``,
       ``N`` means ``Numeric attributes only`` and ``C`` stands for
       ``Categorical attributes only``. It is important to note that the
       categorical attribute sets between ``X`` and ``C`` and the numerical
       attribute sets between ``X`` and ``N`` may differ due to data
       transformations, performed while fitting data into MFE model, enabled
       by, respectively, ``transform_num`` and ``transform_cat`` arguments from
       ``fit`` (MFE method).

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
    def precompute_class_freq(cls, y: t.Optional[np.ndarray] = None,
                              **kwargs) -> t.Dict[str, t.Any]:
        """Precompute each distinct class (absolute) frequencies.

        Parameters
        ----------
        y : :obj:`np.ndarray`, optional
            The target attribute vector.

        kwargs:
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
        :obj:`dict`
            With following precomputed items:
                - ``class_freqs`` (:obj:`np.ndarray`): absolute frequency of
                  each distinct class in ``y``, if ``y`` is not
                  :obj:`None-Type`.
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
                           class_freqs: t.Optional[np.ndarray] = None,
                           **kwargs) -> t.Dict[str, t.Any]:
        """Precompute various values related to Shannon's Entropy.

        Parameters
        ----------
        y : :obj:`np.ndarray`, optional
            The target attribute vector.

        C : :obj:`np.ndarray`, optional
            Categorical attributes from fitted data.

        class_freqs : :obj:`np.ndarray`, optional
            Absolute frequency of each distinct class in ``y``.

        kwargs:
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
        :obj:`dict`
            With following precomputed items:
                - ``class_ent`` (:obj:`float`): Shannon's Entropy of ``y``, if
                  it is not :obj:`NoneType`.
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
            precomp_vals["class_ent"] = MFEInfoTheory.ft_class_ent(
                y, class_freqs=class_freqs)

        if C is not None and C.size and "attr_ent" not in kwargs:
            precomp_vals["attr_ent"] = MFEInfoTheory.ft_attr_ent(C)

        if y is not None and C is not None and C.size:
            if "joint_ent" not in kwargs:
                precomp_vals["joint_ent"] = np.apply_along_axis(
                    func1d=MFEInfoTheory._joint_ent, axis=0, arr=C, vec_y=y)

            if "mut_inf" not in kwargs:
                precomp_vals["mut_inf"] = MFEInfoTheory.ft_mut_inf(
                    C=C,
                    y=y,
                    attr_ent=precomp_vals.get("attr_ent"),
                    class_ent=precomp_vals.get("class_ent"),
                    joint_ent=precomp_vals.get("joint_ent"))

        return precomp_vals

    @classmethod
    def _entropy(cls,
                 values: t.Union[np.ndarray, t.List],
                 value_freqs: t.Optional[np.ndarray] = None) -> float:
        """Calculate Shannon's entropy within array ``values``.

        Check ``ft_attr_ent`` and ``ft_class_ent`` methods for more informa-
        tion.

        Parameters
        ----------
        value_freqs : :obj:`np.ndarray`, optional
            Absolute frequency of each distinct value in ``values``. This
            argument is meant to exploit precomputations.
        """
        if value_freqs is None:
            _, value_freqs = np.unique(values, return_counts=True)

        return scipy.stats.entropy(value_freqs, base=2)

    @classmethod
    def _joint_ent(cls,
                   vec_x: np.ndarray,
                   vec_y: np.ndarray,
                   epsilon: float = 1.0e-10) -> float:
        """Compute joint entropy between vectorx ``x`` and ``y``."""
        joint_prob_mat = pd.crosstab(
            vec_y, vec_x, normalize=True).values + epsilon

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

        Parameters
        ----------
        epsilon : :obj:`float`, optional
            Tiny numeric value to avoid division by zero.
        """
        pij = pd.crosstab(vec_x, vec_y, normalize=True).values + epsilon

        isum = pij.sum(axis=0)
        jsum2 = sum(pij.sum(axis=1)**2.0)

        conc = ((
            (pij**2.0 / isum).sum().sum() - jsum2) / (1.0 - jsum2 + epsilon))

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

        Where ``phi_x`` is a set of all possible distinct values in vector
        ``x`` and P(x = val) is the probability of x assume some value ``val``
        in phi_x.

        Parameters
        ----------
        attr_ent : :obj:`np.ndarray`, optional
            This argument is this method own return value, meant to exploit
            possible attribute entropy precomputations.
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
                     class_ent: t.Optional[np.ndarray] = None,
                     class_freqs: t.Optional[np.ndarray] = None) -> float:
        """Calculates target attribute Shannon's entropy.

        The Shannon's Entropy H of a vector y is defined as:

            H(y) = - sum_{val in phi_y}(P(y = val) * log2(P(y = val))

        Where ``phi_y`` is a set of all possible distinct values in vector
        ``y`` and P(y = val) is the probability of y assume some value ``val``
        in phi_y.

        Parameters
        ----------
        class_ent : :obj:`float`, optional
            Entropy of the target attribute ``y``. Used to explot
            precomputations. If :obj:`NoneType`, this argument is calculated
            using the method ``ft_class_ent``.

        class_freqs : :obj:`np.ndarray`, optional
            Absolute frequency of each distinct class in ``y``. This argument
            is meant to exploit precomputations, used if ``class_ent`` is
            :obj:`NoneType`.
        """
        if class_ent is not None:
            return class_ent

        return MFEInfoTheory._entropy(y, value_freqs=class_freqs)

    @classmethod
    def ft_eq_num_attr(cls,
                       C: np.ndarray,
                       y: np.ndarray,
                       epsilon: float = 1.0e-10,
                       class_ent: t.Optional[np.ndarray] = None,
                       class_freqs: t.Optional[np.ndarray] = None,
                       mut_inf: t.Optional[np.ndarray] = None) -> float:
        """Number of attributes equivalent for a predictive task.

        The attribute equivalence E is defined as:

            E = attr_num * (H(y) / sum_x(MI(x, y)))

        Where H(y) is the Shannon's Entropy of the target attribute and MI(x,y)
        is the Mutual Information between the predictive attribute ``x`` and
        target attribute ``y``.

        Parameters
        ----------
        epsilon : :obj:`float`, optional
            Tiny numeric value to avoid division by zero.

        class_ent : :obj:`float`, optional
            Entropy of the target attribute ``y``. Used to explot
            precomputations. If :obj:`NoneType`, this argument is calculated
            using the method ``ft_class_ent``.

        class_freqs : :obj:`np.ndarray`, optional
            Absolute frequency of each distinct class in ``y``. This argument
            is meant to exploit precomputations, used if ``class_ent`` is
            :obj:`NoneType`.

        mut_inf : :obj:`np.ndarray`, optional
            Values of mutual information between each numeric attribute of
            ``N`` and target ``y``. Similarly, from the argument above, this
            argument purpose is to exploit the precomputations of mutual
            information. If this argument value is :obj:`NoneType`, then it is
            calculated using the method ``ft_mut_int``.
        """
        if class_ent is None:
            class_ent = MFEInfoTheory.ft_class_ent(y, class_freqs=class_freqs)

        if mut_inf is None:
            mut_inf = MFEInfoTheory.ft_mut_inf(C, y)

        _, num_col = C.shape

        return num_col * (class_ent / (epsilon + sum(mut_inf)))

    @classmethod
    def ft_joint_ent(cls,
                     C: np.ndarray,
                     y: np.ndarray,
                     joint_ent: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate the joint entropy between each attribute and class.

        The Joint Entropy H between a predictive attribute x and target attri-
        bute ``y`` is defined as:

            H(x, y) = - sum_{phi_x}(sum_{phi_y}(p_i_j * log2(p_i_j)))

        Where ``phi_x`` and ``phi_y`` are sets of possible distinct values for,
        respectively, ``x`` and ``y`` and ``p_i_j`` is defined as:

            p_i_j = P(x = phi_x_i, y = phi_y_j)

        That is, ``p_i_j`` is the joint probability of ``x`` to assume a speci-
        fic value ``i`` in the set ``phi_x`` simultaneously with ``y`` assuming
        a specific value ``j`` in the set ``phi_y``.

        Parameters
        ----------
        joint_ent : :obj:`np.ndarray`, optional
            This argument is this method own return value, meant to exploit
            possible joint entropy precomputations.
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
                   class_ent: t.Optional[float] = None,
                   joint_ent: t.Optional[np.ndarray] = None,
                   class_freqs: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Mutual information between each attribute in ``C`` and target ``y``.

        The mutual Information MI between an independent attribute ``x`` and
        target attribute ``y`` is defined as:

            MI(x, y) = H(x) + H(y) - H(x, y)

        Where H(x) and H(y) are, respectively, the Shannon's Entropy (see do-
        dumentation of `ft_attr_ent`` or ``ft_class_ent`` for more informati-
        on) for ``x`` and ``y`` and H(x, y) is the joint entropy between ``x``
        and ``y`` (see ``ft_joint_ent`` documentation more details).

        Parameters
        ----------
        attr_ent : :obj:`np.ndarray`, optional
            Values of each attribute entropy in ``N``. This argument purpose is
            to exploit possible precomputations of attribute entropy. If
            :obj:`NoneType`, this argument is calculated using ``ft_attr_ent``
            method.

        mut_inf : :obj:`np.ndarray`, optional
            This argument is this method own return value, meant to exploit
            possible mutual information precomputations.

        class_ent : :obj:`float`, optional
            Entropy of the target attribute ``y``. Used to explot
            precomputations. If :obj:`NoneType`, this argument is calculated
            using the method ``ft_class_ent``.

        joint_ent : :obj:`np.ndarray`, optional
            Joint entropy between each independent attribute in ``N`` and
            target attribute ``y``. If :obj:`NoneType`, this argument is
            calculated using the method ``ft_joint_ent``.
        """
        if mut_inf is not None:
            return mut_inf

        if class_ent is None:
            class_ent = MFEInfoTheory.ft_class_ent(y, class_freqs=class_freqs)

        if attr_ent is None:
            attr_ent = MFEInfoTheory.ft_attr_ent(C)

        if joint_ent is None:
            joint_ent = MFEInfoTheory.ft_joint_ent(C, y)

        return attr_ent + class_ent - joint_ent

    @classmethod
    def ft_ns_ratio(cls,
                    C: np.ndarray,
                    y: np.ndarray,
                    epsilon: float = 1.0e-10,
                    attr_ent: t.Optional[np.ndarray] = None,
                    mut_inf: t.Optional[np.ndarray] = None) -> float:
        """Compute the noisiness of attributes.

        Let ``y`` be a target attribute and ``x`` one predictive attribute in
        a dataset ``N``. Noisiness ``N`` is defined as:

            N = (sum_x(attr_entropy(x)) - sum_x(MI(x, y))) / sum_x(MI(x, y))

        where MI(x, y) is the mutual information between target attribute ``y``
        and predictive attribute ``x``, and all ``sum`` is performed for all
        distinct ``x`` in ``N``.

        Parameters
        ----------
        epsilon : :obj:`float`, optional
            Tiny numeric value to avoid division by zero.

        attr_ent : :obj:`np.ndarray`, optional
            Values of each attribute entropy in ``N``. This argument purpose is
            to exploit possible precomputations of attribute entropy. If
            :obj:`NoneType`, this argument is calculated using ``ft_attr_ent``
            method.

        mut_inf : :obj:`np.ndarray`, optional
            Values of mutual information between each numeric attribute of
            ``N`` and target ``y``. Similarly, from the argument above, this
            argument purpose is to exploit the precomputations of mutual
            information. If this argument value is :obj:`NoneType`, then it is
            calculated using the method ``ft_mut_int``.
        """
        if attr_ent is None:
            attr_ent = MFEInfoTheory.ft_attr_ent(C)

        if mut_inf is None:
            mut_inf = MFEInfoTheory.ft_mut_inf(C, y)

        ent_attr = sum(attr_ent)
        mut_inf = sum(mut_inf)

        return (ent_attr - mut_inf) / (epsilon + mut_inf)
