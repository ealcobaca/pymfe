"""Module dedicated to extraction of Relative Landmarking Metafeatures.
"""

import typing as t
import numpy as np
import pymfe.landmarking as landmarking


class MFERelative(landmarking.MFELandmarking):
    """Keep methods for metafeatures of ``relative`` group.

    The convention adopted for metafeature extraction related methods is to
    always start with ``ft_`` prefix to allow automatic method detection. This
    prefix is predefined within ``_internal`` module. Since this group is based
    on landmarking, the metafeatures should be implemented on
    ``MFELandmarking`` class.

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
    precomputation or feature extraction method of module ``relative``).
    """

    @classmethod
    def precompute_relative_class(cls, N: np.ndarray, y: np.ndarray,
                                  size: float, folds: int,
                                  random_state: t.Optional[int],
                                  **kwargs) -> t.Dict[str, t.Any]:
        """Precompute subsampling and k-fold cross validation strategy.

        Parameters
        ----------
        N : :obj:`np.ndarray`, optional
            Attributes from fitted data.

        y : :obj:`np.ndarray`, optional
            Target attribute from fitted data.

        size : :obj: `float`
            The percentage of examples subsampled.

        folds : :obj: `int`
            Number of folds to k-fold cross validation.

        random_state : :obj:`int`, optional
            If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random
            number generator; If None, the random number generator is the
            RandomState instance used by np.random.

        kwargs:
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
        :obj:`dict`
            With following precomputed items:
                - ``skf`` (:obj:`StratifiedKFold`): Stratified K-Folds
                  cross-validator. Provides train/test indices to split data in
                  train/test sets.
        """

        prepcomp_vals = {}

        if N is not None and y is not None\
           and not {"skf"}.issubset(kwargs):

            size = int(size*N.shape[0])
            idx = np.random.choice(N.shape[0], size)
            N = N[idx, :]
            y = y[idx]

            skf = landmarking.StratifiedKFold(n_splits=folds,
                                              random_state=random_state)
            prepcomp_vals["skf"] = skf

        return prepcomp_vals
