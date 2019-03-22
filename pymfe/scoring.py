"""Scoring module.
"""
# import typing as t
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ Calculates the accuracy of a classification model.
    """
    return accuracy_score(y_true, y_pred)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ Calculates the balanced accuracy of a classification model.
    """
    return balanced_accuracy_score(y_true, y_pred)


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ Calculates the F1-score of a classification model.
    """
    return f1_score(y_true, y_pred, average='micro')


def kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ Calculates the Kappa-score of a classification model.
    """
    raise NotImplementedError('The "kappa" score was not implemented.')


def auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ Calculates the AUC of a classification model.
    """
    raise NotImplementedError('The "auc" score was not implemented.')
