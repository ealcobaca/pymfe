"""Module dedicated to extraction of Complexity Metafeatures."""

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
    def precompute_foo_itemset(cls,
                               N: np.ndarray,
                               **kwargs) -> t.Dict[str, t.Any]:
        """Precompute PCA to Tx complexit measures.

        Parameters
        ----------
        N : :obj:`np.ndarray`, optional
            Attributes from fitted data.

        **kwargs
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
        :obj:`dict`
            With following precomputed items:
                - ``m`` (:obj:`int`): Number of features.
                - ``m_`` (:obj:`int`):  Number of features after PCA with 0.95.
                - ``n`` (:obj:`int`): Number of examples.
        """
        prepcomp_vals = {}
        return prepcomp_vals

    @staticmethod
    def _array_to_binary(col):
        values = np.unique(col)
        res = np.zeros((col.shape[0], values.shape[0])).astype(bool)
        for i, val in enumerate(values):
            res[:, i] = col == val
        return res

    @staticmethod
    def _matrix_to_binary(C_: np.array):
        return [MFEItemset._array_to_binary(col) for col in C_.T]

    @classmethod
    def ft_one_itemset(cls,
                       C: np.ndarray,
                       ) -> np.ndarray:
        """TODO.
        """
        B = MFEItemset._matrix_to_binary(C)
        B = np.concatenate(B, axis=1)
        oneitem_by_attr = np.sum(B, axis=0) / C.shape[0]

        return oneitem_by_attr
