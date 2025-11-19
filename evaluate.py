# evaluate.py
"""
Evaluation metrics for forecasting:
 - mae(y_true, y_pred)
 - rmse(y_true, y_pred)
 - mape(y_true, y_pred)  (returns percent, e.g. 2.5 means 2.5%)
 - directional_accuracy(y_true, y_pred)  (fraction of correct up/down signs)
"""
from typing import Sequence
import numpy as np


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Mean Absolute Percentage Error (as percentage). If y_true contains zeros, uses safe denominator.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, 1e-9, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def directional_accuracy(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Fraction of times the predicted direction (delta sign) matches actual direction.
    Uses one-step deltas: sign(y_t - y_{t-1}).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    # align lengths
    n = min(len(true_dir), len(pred_dir))
    if n == 0:
        return 0.0
    return float(np.mean(true_dir[:n] == pred_dir[:n]))
