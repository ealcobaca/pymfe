"""Test module for General class metafeatures."""

import pytest
import numpy as np

from pymfe.scoring import accuracy
from pymfe.scoring import balanced_accuracy
from pymfe.scoring import f1


def test_accuracy():
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert accuracy(y_true, y_pred) == 1.0

    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    assert accuracy(y_true, y_pred) == 0.0

    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert accuracy(y_true, y_pred) == 0.5

    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0])
    assert accuracy(y_true, y_pred) == 0.6


def test_balanced_accuracy():
    y_true = np.array([2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert np.isclose(balanced_accuracy(y_true, y_pred), 1.0)

    y_true = np.array([2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    assert np.isclose(balanced_accuracy(y_true, y_pred), 0.0)

    y_true = np.array([2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert np.isclose(balanced_accuracy(y_true, y_pred), 1 / 3)


def test_f1():
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert np.isclose(f1(y_true, y_pred), 1.0)

    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    assert np.isclose(f1(y_true, y_pred), 0.0)

    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert np.isclose(f1(y_true, y_pred), 1 / 3)

    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0])
    assert np.isclose(f1(y_true, y_pred), 0.6)

    y_true = np.array([2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert np.isclose(f1(y_true, y_pred), 1 / 6)
