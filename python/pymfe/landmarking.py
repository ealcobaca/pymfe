"""Module dedicated to extraction of Landmarking Metafeatures.

Todo:
    * Implement metafeatures.
    * Improve documentation.
"""

import typing as t
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np



class MFELandmarking:
    """To do this documentation."""

    @classmethod
    def precompute_landmarking_class(cls, X: np.array, y: np.array,
                                     # folds: int, score: str, random_state: int,
                                     **kwargs) -> t.Dict[str, t.Any]:
        prepcomp_vals = {}

        if X is not None and y is not None\
           and not {"model", "table"}.issubset(kwargs):
            skf = StratifiedKFold(n_splits=2)
            prepcomp_vals["skf"] = skf

        return prepcomp_vals

    @classmethod
    def ft_best_node(cls, X, y, skf):
        clf = DecisionTreeClassifier(max_depth=1)
        result = cross_val_score(clf, X, y,
                                 scoring="accuracy", cv=10)
        # result = []
        # for train_index, test_index in skf.split(X, y):
        #     model = DecisionTreeClassifier(max_depth=1)
        #     model.fit(X[train_index], y[train_index])
        #     pred = model.predict(X[test_index])
        #     result.append(accuracy_score(y[test_index], pred))

        return np.array(result)

    @classmethod
    def ft_elite_nn(cls):
        pass

    @classmethod
    def ft_linear_discr(cls):
        pass

    @classmethod
    def ft_naive_bayes(cls):
        pass

    @classmethod
    def ft_one_nn(cls):
        pass

    @classmethod
    def ft_random_node(cls):
        pass

    @classmethod
    def worst_node(cls):
        pass

    @classmethod
    def ft_test_landmarking(cls):
        """To do."""
        return 0.0
