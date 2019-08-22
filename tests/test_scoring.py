"""Test module for General class metafeatures."""

import pytest
import numpy as np
import sklearn.metrics


def test_accuracy():
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert sklearn.metrics.balanced_accuracy_score(y_true, y_pred) == 1.0

    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    assert sklearn.metrics.balanced_accuracy_score(y_true, y_pred) == 0.0

    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert sklearn.metrics.balanced_accuracy_score(y_true, y_pred) == 0.5

    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0])
    assert sklearn.metrics.balanced_accuracy_score(y_true, y_pred) == 0.6


def test_balanced_accuracy():
    y_true = np.array([2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert sklearn.metrics.balanced_accuracy_score(y_true, y_pred) == 1.0

    y_true = np.array([2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    assert sklearn.metrics.balanced_accuracy_score(y_true, y_pred) == 0.0

    y_true = np.array([2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert sklearn.metrics.balanced_accuracy_score(y_true, y_pred) == 1/3


def test_f1():
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert sklearn.metrics.f1_score(y_true, y_pred, average='weighted') == 1.0

    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    assert sklearn.metrics.f1_score(y_true, y_pred, average='weighted') == 0.0

    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert sklearn.metrics.f1_score(y_true, y_pred, average='weighted') == 1/3

    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0])
    assert sklearn.metrics.f1_score(y_true, y_pred, average='weighted') == 0.6

    y_true = np.array([2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert sklearn.metrics.f1_score(y_true, y_pred, average='weighted') == 1/6
