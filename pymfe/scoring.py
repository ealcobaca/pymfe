"""Scoring module."""
import numpy as np
import sklearn.metrics


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the accuracy of a classification model."""
    return sklearn.metrics.accuracy_score(y_true, y_pred)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the balanced accuracy of a classification model."""
    return sklearn.metrics.balanced_accuracy_score(y_true, y_pred)


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the F1-score of a classification model."""
    return sklearn.metrics.f1_score(y_true, y_pred, average="weighted")


def kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the Kappa-score of a classification model."""
    raise NotImplementedError('The "kappa" score was not implemented.')


def auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the AUC of a classification model."""
    raise NotImplementedError('The "auc" score was not implemented.')
