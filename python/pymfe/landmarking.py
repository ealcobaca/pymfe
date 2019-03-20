"""Module dedicated to extraction of Landmarking Metafeatures.

Todo:
    * Implement metafeatures.
    * Improve documentation.
"""

import typing as t
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np



class MFELandmarking:
    """To do this documentation."""

    @classmethod
    def precompute_landmarking_class(cls, X: np.ndarray, y: np.ndarray,
                                     folds: int, random_state: t.Optional[int],
                                     **kwargs) -> t.Dict[str, t.Any]:
        prepcomp_vals = {}

        if X is not None and y is not None\
           and not {"model", "table"}.issubset(kwargs):
            skf = StratifiedKFold(n_splits=folds, random_state=random_state)
            prepcomp_vals["skf"] = skf

        return prepcomp_vals

    @classmethod
    def importance(cls, X: np.ndarray, y: np.ndarray,
                   random_state: t.Optional[int]) -> np.ndarray:
        clf = DecisionTreeClassifier(random_state=random_state).fit(X, y)
        return np.argsort(clf.feature_importances_)

    @classmethod
    def ft_best_node(cls, X: np.ndarray, y: np.ndarray,
                     skf, score, random_state: t.Optional[int]) -> np.ndarray:
        result = []
        for train_index, test_index in skf.split(X, y):
            # importance = MFELandmarking.importance(X[train_index],
            #                                        y[train_index],
            #                                        random_state)
            model = DecisionTreeClassifier(max_depth=1,
                                           random_state=random_state)
            X_train = X[train_index, :]
            X_test = X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result.append(score(y_test, pred))

        return np.ndarray(result)

    @classmethod
    def ft_random_node(cls, X, y, skf, score, random_state):
        result = []
        for train_index, test_index in skf.split(X, y):
            attr = np.random.randint(0, X.shape[1], size=(1,))
            model = DecisionTreeClassifier(max_depth=1,
                                           random_state=random_state)
            X_train = X[train_index, :][:, attr]
            X_test = X[test_index, :][:, attr]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result.append(score(y_test, pred))

        return np.ndarray(result)

    @classmethod
    def ft_worst_node(cls, X, y, skf, score, random_state):
        result = []
        for train_index, test_index in skf.split(X, y):
            importance = MFELandmarking.importance(X[train_index],
                                                   y[train_index],
                                                   random_state)
            model = DecisionTreeClassifier(max_depth=1,
                                           random_state=random_state)
            X_train = X[train_index, :][:, [importance[0]]]
            X_test = X[test_index, :][:, [importance[0]]]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result.append(score(y_test, pred))

        return np.ndarray(result)

    @classmethod
    def ft_elite_nn(cls, X, y, skf, score, random_state):
        result = []
        for train_index, test_index in skf.split(X, y):
            importance = MFELandmarking.importance(X[train_index],
                                                   y[train_index],
                                                   random_state)
            model = KNeighborsClassifier(n_neighbors=1)
            X_train = X[train_index, :][:, [importance[-1]]]
            X_test = X[test_index, :][:, [importance[-1]]]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result.append(score(y_test, pred))

        return np.ndarray(result)

    @classmethod
    def ft_linear_discr(cls, X, y, skf, score):
        result = []
        for train_index, test_index in skf.split(X, y):
            model = LinearDiscriminantAnalysis()
            X_train = X[train_index, :]
            X_test = X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result.append(score(y_test, pred))

        return np.ndarray(result)

    @classmethod
    def ft_naive_bayes(cls, X, y, skf, score):
        result = []
        for train_index, test_index in skf.split(X, y):
            model = GaussianNB()
            X_train = X[train_index, :]
            X_test = X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result.append(score(y_test, pred))

        return np.ndarray(result)

    @classmethod
    def ft_one_nn(cls, X, y, skf, score):
        result = []
        for train_index, test_index in skf.split(X, y):
            model = KNeighborsClassifier(n_neighbors=1)
            X_train = X[train_index, :]
            X_test = X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result.append(score(y_test, pred))

        return np.ndarray(result)
