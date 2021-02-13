"""Module dedicated to extraction of itemset metafeatures."""

import typing as t

import numpy as np


class MFEItemset:
    """Keep methods for metafeatures of ``Itemset`` group.

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
       value or a generic List (preferably a :obj:`np.ndarray`) type with
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
    def precompute_binary_matrix(
        cls, C: t.Optional[np.ndarray], **kwargs
    ) -> t.Dict[str, t.Any]:
        """Precompute the binary representation of attributes.

        Parameters
        ----------
        C : :obj:`np.ndarray`, optional
            Categorical fitted data.

        **kwargs
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
        :obj:`dict`
            With following precomputed items:
                - ``itemset_binary_matrix`` (:obj:`list`): Binary
                  representation of the attributes.
        """
        precomp_vals = {}

        if C is not None and "itemset_binary_matrix" not in kwargs:
            itemset_binary_matrix = cls._matrix_to_binary(C)
            precomp_vals["itemset_binary_matrix"] = itemset_binary_matrix

        return precomp_vals

    @staticmethod
    def _array_to_binary(array: np.ndarray) -> np.ndarray:
        """Convert an array to its binary representation."""
        values = np.unique(array)
        res = np.zeros((array.shape[0], values.shape[0])).astype(bool)
        for i, val in enumerate(values):
            res[:, i] = array == val
        return res

    @classmethod
    def _matrix_to_binary(cls, C: np.ndarray) -> t.List[np.ndarray]:
        """Convert an matrix to its binary representation."""
        return [cls._array_to_binary(col) for col in C.T]

    @classmethod
    def ft_two_itemset(
        cls,
        C: np.ndarray,
        itemset_binary_matrix: t.List[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the two itemset meta-feature.

        The two-item set meta-feature can be seen as the correlation
        information of each one attributes value pairs in binary
        format.

        Parameters
        ----------
        C : :obj:`np.ndarray`
            Categorical fitted data.

        itemset_binary_matrix : :obj:`list` of :obj:`np.ndarray`, optional
            Binary representation of the attributes. Each list value has a
            binary representation of each attributes in the dataset.

        Returns
        -------
        :obj:`np.ndarray`
            An array with the two-item value for each attribute.

        References
        ----------
        .. [1] Song, Q., Wang, G., & Wang, C. (2012). Automatic recommendation
           of classification algorithms based on data set characteristics.
           Pattern recognition, 45(7), 2672-2689.
        """
        if itemset_binary_matrix is None:
            sub_dic = cls.precompute_binary_matrix(C)
            itemset_binary_matrix = sub_dic["itemset_binary_matrix"]

        B = itemset_binary_matrix

        result = []  # type: t.List[float]
        while B:
            Bi = B[0]
            del B[0]
            for Bj in B:
                aux = [
                    np.sum(np.logical_xor(i, j)) for i in Bi.T for j in Bj.T
                ]
                result += aux

        twoitem_by_attr = np.array(result) / C.shape[0]

        return twoitem_by_attr

    @classmethod
    def ft_one_itemset(
        cls,
        C: np.ndarray,
        itemset_binary_matrix: t.List[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the one itemset meta-feature.

        The one itemset is the individual frequency of each attribute
        in binary format.

        Parameters
        ----------
        C : :obj:`np.ndarray`
            Categorical fitted data.

        itemset_binary_matrix : :obj:`list` of :obj:`np.ndarray`, optional
            Binary representation of the attributes. Each list value has a
            binary representation of each attributes in the dataset.

        Returns
        -------
        :obj:`np.ndarray`
            An array with the one-item value for each attribute.

        References
        ----------
        .. [1] Song, Q., Wang, G., & Wang, C. (2012). Automatic recommendation
           of classification algorithms based on data set characteristics.
           Pattern recognition, 45(7), 2672-2689.
        """
        if itemset_binary_matrix is None:
            sub_dic = cls.precompute_binary_matrix(C)
            itemset_binary_matrix = sub_dic["itemset_binary_matrix"]

        B = itemset_binary_matrix
        B = np.concatenate(B, axis=1)

        oneitem_by_attr = np.sum(B, axis=0) / C.shape[0]

        return oneitem_by_attr
