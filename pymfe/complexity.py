"""Module dedicated to extraction of Complexity Metafeatures."""

import typing as t

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

        # I think we could improve this part of the code
        # an alternative is given above
        # idx_class = np.zeros(N.shape[0], idx_classes.shape[0])
        # idx_class[np.arrange(y.shape[0]), y_idx] = 1
        y_idx_all_classes = [np.equal(y_idx, idx)
                             for idx in range(classes.shape[0])]

        # get min max for all columns by class
        # line.... --> class
        # columns. --> features
        min_cls = [np.min(N[feature, :], axis=0)
                   for feature in y_idx_all_classes]
        max_cls = [np.max(N[feature, :], axis=0)
                   for feature in y_idx_all_classes]
        min_cls = np.array(min_cls)
        max_cls = np.array(max_cls)

        if N is not None and y is not None:
            if "minmax" not in kwargs:
                minmax = np.max(max_cls, axis=0)
                prepcomp_vals["minmax"] = minmax
            if "maxmin" not in kwargs:
                maxmin = np.max(min_cls)
                prepcomp_vals["maxmin"] = maxmin

        print(precompute_vals)
        return prepcomp_vals

    @staticmethod
    def _compute_f3(N_: np.ndarray,
                    y_: np.ndarray,
                    minmax_: np.ndarray,
                    maxmin_: np.ndarray
                    ) -> np.ndarray:

         # True if the example is in the overlapping region
        overlapped_region_by_feature = np.logical_and(N_ > maxmin, N_ < maxmin)
        n_fi = np.sum(overlapped_region_by_feature, axis=0)
        min_n_fi = np.min(n_fi)
        idx_min = np.argmin(min_n_fi)

        return idx_min, min_n_fi, overlapped_region_by_feature

    @classmethod
    def ft_f3(cls,
              N: np.ndarray,
              y: np.ndarray,
              minmax: np.ndarray,
              maxmin: np.ndarray
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
        idx_min, min_n_fi, _ = cls._compute_f3(N, y, minmax, maxmin)
        f3 = min_n_fi[idx_min] / N.shape[0]

        return f3

    @classmethod
    def ft_f4(cls,
              N: np.ndarray,
              y: np.ndarray,
              minmax: np.ndarray,
              maxmin: np.ndarray
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

        N_ = N
        y_ = y
        minmax_ = minmax
        maxmin_ = maxmin
        for i in range(N.shape[1]):
            # True if the example is in the overlapping region
            idx_min, _, overlapped_region_by_feature = cls._compute_f3(
                N, y, minmax, maxmin)

            # boolean that if True, this example is in the overlapping region
            overlapped_region = overlapped_region_by_feature[:, idx_min]
            # removing the most efficient feature
            N_ = np.delete(N_, idx_min, axis=0)
            minmax_ = minmax_[idx_min]
            maxmin_ = maxmin_[idx_min]
            # removing the non overlapped features
            N_ = N_[overlapped_region, :]
            y_ = y_[overlapped_region, :]

            if N.shape[0] == 0:
                no_overlap = True
                break

        f4 = 0 if no_overlap else N_.shape[0]/N.shape[0]

        return f4
