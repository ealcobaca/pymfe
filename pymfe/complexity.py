"""Module dedicated to extraction of Complexity Metafeatures."""

import typing as t
import itertools
import numpy as np


class MFEComplexity:
    """Keep methods for metafeatures of ``landmarking`` group.

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
    def precompute_fx(cls,
                      N: np.ndarray,
                      y: np.ndarray,
                      **kwargs
                      ) -> t.Dict[str, t.Any]:

        prepcomp_vals = {}

        classes, idx_classes, y_idx, class_freqs = np.unique(
            y, return_index=True, return_inverse=True, return_counts=True)

        if (y is not None and
                "ovo_comb" not in kwargs and
                "cls_index" not in kwargs and
                "cls_amount" not in kwargs):
            n_classe = 2
            cls_index = [np.equal(y_idx, i)
                         for i in range(classes.shape[0])]
            cls_amount = [np.sum(aux) for aux in cls_index]
            ovo_comb = list(itertools.combinations(range(classes.shape[0]), 2))
            prepcomp_vals["ovo_comb"] = ovo_comb
            prepcomp_vals["cls_index"] = cls_index
            prepcomp_vals["cls_amount"] = cls_amount
        return prepcomp_vals

    @staticmethod
    def _minmax(N: np.ndarray,
                class1:np.ndarray,
                class2:np.ndarray
                ) -> np.ndarray:
        min_cls = np.zeros((2, N.shape[1]))
        min_cls[0, :] = np.max(N[class1], axis=0)
        min_cls[1, :] = np.max(N[class2], axis=0)
        aux = np.min(min_cls, axis=0)
        return aux

    @staticmethod
    def _maxmin(N: np.ndarray,
                class1: np.ndarray,
                class2: np.ndarray
                ) -> np.ndarray:
        max_cls = np.zeros((2, N.shape[1]))
        max_cls[0, :] = np.min(N[class1], axis=0)
        max_cls[1, :] = np.min(N[class2], axis=0)
        aux = np.max(max_cls, axis=0)
        return aux

    @staticmethod
    def _compute_f3(N_: np.ndarray,
                    minmax_: np.ndarray,
                    maxmin_: np.ndarray
                    ) -> np.ndarray:

        # True if the example is in the overlapping region
        # Should be > and < instead of >= and <= ?
        overlapped_region_by_feature = np.logical_and(
            N_ >= maxmin_, N_ <= minmax_)

        n_fi = np.sum(overlapped_region_by_feature, axis=0)
        idx_min = np.argmin(n_fi)

        return idx_min, n_fi, overlapped_region_by_feature

    @classmethod
    def ft_f3(cls,
              N: np.ndarray,
              ovo_comb: np.ndarray,
              cls_index: np.ndarray,
              cls_amount: np.ndarray,
              ) -> np.ndarray:
        """Performance of a the best single decision tree node.

        Construct a single decision tree node model induced by the most
        informative attribute to establish the linear separability.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        Returns
        -------
        :obj:`np.ndarray`
            The performance of each fold.
        """
        f3 = []
        for idx1, idx2 in ovo_comb:
            idx_min, n_fi, _ = cls._compute_f3(
                N,
                cls._minmax(N, cls_index[idx1], cls_index[idx2]),
                cls._maxmin(N, cls_index[idx1], cls_index[idx2])
            )
            f3.append(n_fi[idx_min] / (cls_amount[idx1] + cls_amount[idx2]))

        return np.mean(f3)

    @classmethod
    def ft_f4(cls,
              N: np.ndarray,
              ovo_comb,
              cls_index,
              cls_amount,
              ) -> np.ndarray:
        """Performance of a the best single decision tree node.

        Construct a single decision tree node model induced by the most
        informative attribute to establish the linear separability.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        Returns
        -------
        :obj:`np.ndarray`
            The performance of each fold.
        """

        f4 = []
        for idx1, idx2 in ovo_comb:
            no_overlap = False
            aux = 0

            y_class1 = cls_index[idx1]
            y_class2 = cls_index[idx2]
            sub_set = np.logical_or(y_class1, y_class2)
            y_class1 = y_class1[sub_set]
            y_class2 = y_class2[sub_set]
            N_ = N[sub_set, :]
            # N_ = N[np.logical_or(y_class1, y_class2),:]

            for i in range(N.shape[1]):
                # True if the example is in the overlapping region
                idx_min, _, overlapped_region_by_feature = cls._compute_f3(
                    N_,
                    cls._minmax(N_, y_class1, y_class2),
                    cls._maxmin(N_, y_class1, y_class2)
                )

                # boolean that if True, this example is in the overlapping region
                overlapped_region = overlapped_region_by_feature[:, idx_min]

                # removing the non overlapped features
                N_ = N_[overlapped_region, :]
                y_class1 = y_class1[overlapped_region]
                y_class2 = y_class2[overlapped_region]

                if N_.shape[0] == 0:
                    aux = 0
                    break
                else:
                    aux = N_.shape[0]

                # removing the most efficient feature
                N_ = np.delete(N_, idx_min, axis=1)

            f4.append(aux/(cls_amount[idx1] + cls_amount[idx2]))

        return np.mean(f4)
