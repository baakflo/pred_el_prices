"""Point-forecast metrics following the epftoolbox conventions (Lago et al. 2021).

Kept for benchmark comparability; probabilistic scores live elsewhere.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE in percent, epftoolbox definition."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))))


def naive_forecast(prices: pd.Series, kind: str = "mixed") -> pd.Series:
    """EPF naive baselines on an hourly series (first 7 days yield NaN).

    kind="mixed": Tue-Fri copy yesterday, Mon/Sat/Sun copy last week (the
    "standard" naive). kind="weekly": always copy last week — this is the
    baseline behind the published rMAE tables in Lago et al. 2021.
    """
    lag7 = prices.shift(24 * 7)
    if kind == "weekly":
        return lag7
    lag1 = prices.shift(24)
    use_weekly = prices.index.dayofweek.isin([0, 5, 6])  # Mon, Sat, Sun
    return lag7.where(use_weekly, lag1)


def rmae_with_history(
    history: pd.Series, test: pd.Series, y_pred: np.ndarray, kind: str = "weekly"
) -> float:
    """rMAE where the naive baseline may reach 7 days before the test period.

    `history` must contain hourly prices covering at least 7 days before
    `test.index[0]` up to `test.index[-1]`. Default kind matches the paper's
    published tables (weekly persistence).
    """
    naive = naive_forecast(history, kind).reindex(test.index)
    return mae(test.values, y_pred) / mae(test.values, naive.values)


def mae_by_hour(actual: pd.Series, pred: pd.Series) -> dict[int, float]:
    """MAE per UTC hour-of-day (0..23), keyed by hour."""
    err = (actual - pred).abs()
    by_hour = err.groupby(actual.index.hour).mean()
    return {int(h): float(v) for h, v in by_hour.items()}


def negative_hour_metrics(actual: pd.Series, pred: pd.Series) -> dict:
    """Diagnostics for how well negative-price hours are captured.

    Ratios/medians are NaN (never raise) when the underlying count is zero.
    `depth_ratio` is the median predicted price over the median actual price,
    restricted to hours where both actual and pred are negative.
    """
    neg_actual = actual < 0
    neg_pred = pred < 0
    joint_neg = neg_actual & neg_pred
    n_neg_actual = int(neg_actual.sum())
    n_neg_pred = int(neg_pred.sum())
    sign_recall = float(joint_neg.sum() / n_neg_actual) if n_neg_actual > 0 else float("nan")
    sign_precision = float(joint_neg.sum() / n_neg_pred) if n_neg_pred > 0 else float("nan")
    if joint_neg.any():
        median_pred_joint_neg = float(pred[joint_neg].median())
        median_actual_joint_neg = float(actual[joint_neg].median())
        depth_ratio = median_pred_joint_neg / median_actual_joint_neg
    else:
        median_pred_joint_neg = float("nan")
        median_actual_joint_neg = float("nan")
        depth_ratio = float("nan")
    nonneg = actual >= 0
    MAE_nonneg_actual = (
        float((actual[nonneg] - pred[nonneg]).abs().mean()) if nonneg.any() else float("nan")
    )
    return {
        "n_neg_actual": n_neg_actual,
        "n_neg_pred": n_neg_pred,
        "sign_recall": sign_recall,
        "sign_precision": sign_precision,
        "median_pred_joint_neg": median_pred_joint_neg,
        "median_actual_joint_neg": median_actual_joint_neg,
        "depth_ratio": depth_ratio,
        "MAE_nonneg_actual": MAE_nonneg_actual,
    }


def dm_test(actual: pd.Series, pred_a: pd.Series, pred_b: pd.Series) -> tuple[float, float]:
    """Diebold-Mariano test, epftoolbox "multivariate" convention (Lago et al.
    2021, epftoolbox.evaluation.DM with norm=1, version='multivariate').

    Per UTC day t, d_t = sum_h |actual - pred_a| - sum_h |actual - pred_b|
    over the 24 hours of that day (scaling by mean instead of sum, as
    epftoolbox does, would give the same statistic since it cancels).
    DM = mean(d) / sqrt(var(d, ddof=0) / N), one-sided p = 1 - Phi(DM) via
    scipy.stats.norm (epftoolbox uses ddof=0 for the variance).

    A small p-value means B is significantly more accurate than A. Requires a
    sorted UTC-hourly index made of whole days: each consecutive block of 24
    rows covers hours 0..23 of a single calendar day, in order (days
    themselves need not be adjacent); raises ValueError otherwise. If d has
    zero variance (e.g. pred_a == pred_b), the statistic is undefined; this
    degenerate case returns (0.0, 0.5) (no evidence of a difference) rather
    than raising or dividing by zero.
    """
    idx = actual.index
    if not (pred_a.index.equals(idx) and pred_b.index.equals(idx)):
        raise ValueError("actual, pred_a, pred_b must share the same index")
    if len(idx) == 0 or len(idx) % 24 != 0:
        raise ValueError(f"dm_test requires whole UTC days (24h blocks); got {len(idx)} hours")
    if not idx.is_monotonic_increasing:
        raise ValueError("dm_test requires a sorted index")
    hours = idx.hour.to_numpy().reshape(-1, 24)
    days = idx.normalize().to_numpy().reshape(-1, 24)
    if not (hours == np.arange(24)).all() or not (days == days[:, [0]]).all():
        raise ValueError(
            "dm_test requires whole UTC days (hours 0..23, one calendar day per block)"
        )

    err_a = (actual - pred_a).abs().to_numpy().reshape(-1, 24)
    err_b = (actual - pred_b).abs().to_numpy().reshape(-1, 24)
    d = err_a.sum(axis=1) - err_b.sum(axis=1)
    n = len(d)
    var_d = d.var(ddof=0)
    if var_d == 0:
        return 0.0, 0.5
    dm_stat = d.mean() / np.sqrt(var_d / n)
    p_value = 1 - norm.cdf(dm_stat)
    return float(dm_stat), float(p_value)
