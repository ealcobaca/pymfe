"""Keeps generic utility functions."""
import typing as t

import numpy as np


def calc_cls_inds(
    y: np.ndarray, classes: t.Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute the ``cls_inds`` variable.

    The ``cls_inds`` variable is a boolean array which marks with
    True value whether the instance belongs to each class. Each
    distinct class is represented by a row, and each instance is
    represented by a column.
    """
    if classes is None:
        classes = np.unique(y)

    cls_inds = np.array(
        [np.equal(y, cur_cls) for cur_cls in classes], dtype=bool
    )

    return cls_inds
