"""Point-forecast metrics following the epftoolbox conventions (Lago et al. 2021).

Kept for benchmark comparability; probabilistic scores live elsewhere.
"""

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE in percent, epftoolbox definition."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))))


def naive_forecast(prices: pd.Series) -> pd.Series:
    """Standard EPF naive: Tue-Fri copy yesterday, Mon/Sat/Sun copy last week.

    `prices` is an hourly series; the first 7 days yield NaN.
    """
    lag1 = prices.shift(24)
    lag7 = prices.shift(24 * 7)
    use_weekly = prices.index.dayofweek.isin([0, 5, 6])  # Mon, Sat, Sun
    return lag7.where(use_weekly, lag1)


def rmae_with_history(history: pd.Series, test: pd.Series, y_pred: np.ndarray) -> float:
    """rMAE where the naive baseline may reach 7 days before the test period.

    `history` must contain hourly prices covering at least 7 days before
    `test.index[0]` up to `test.index[-1]`.
    """
    naive = naive_forecast(history).reindex(test.index)
    return mae(test.values, y_pred) / mae(test.values, naive.values)
