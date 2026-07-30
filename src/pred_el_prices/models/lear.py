"""LEAR — LASSO-estimated autoregressive model, epftoolbox-faithful reimplementation.

Spec (Lago et al. 2021, and the epftoolbox reference code):
- One linear model per delivery hour h, recalibrated every day on a rolling
  window of `calibration_window` days.
- Features per day d (247 for two exogenous series):
  prices of d-1, d-2, d-3, d-7 (4 x 24), each exogenous series at d, d-1, d-7
  (3 x 24 each), 7 day-of-week dummies.
- "Invariant" scaling fit on the training window: (x - median) / mad, then
  asinh; dummies unscaled. mad follows statsmodels' default normalization
  (median absolute deviation / 0.6745).
- Per hour: LassoLarsIC(criterion="aic") picks alpha, Lasso(alpha) fits,
  prediction is inverse-transformed.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoLarsIC

PRICE_LAG_DAYS = [1, 2, 3, 7]
EXOG_LAG_DAYS = [0, 1, 7]


class InvariantScaler:
    """Median/MAD normalization followed by asinh (epftoolbox 'Invariant')."""

    def fit(self, x: np.ndarray) -> "InvariantScaler":
        self.median = np.median(x, axis=0)
        self.mad = np.median(np.abs(x - self.median), axis=0) / 0.6745
        self.mad = np.where(self.mad == 0, 1.0, self.mad)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.arcsinh((x - self.median) / self.mad)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return np.sinh(x) * self.mad + self.median


def build_xy(prices: np.ndarray, exog: np.ndarray, dayofweek: np.ndarray):
    """Day-indexed design matrix and 24-wide target.

    prices: (n_days, 24); exog: (n_days, 24, n_exog); dayofweek: (n_days,).
    Rows for the first 7 days are dropped (lag burn-in). Returns X, Y, and the
    row->day offset (7).
    """
    n_days = prices.shape[0]
    n_exog = exog.shape[2]
    rows = range(7, n_days)
    blocks = []
    for lag in PRICE_LAG_DAYS:
        blocks.append(prices[[d - lag for d in rows], :])
    for j in range(n_exog):
        for lag in EXOG_LAG_DAYS:
            blocks.append(exog[[d - lag for d in rows], :, j])
    dummies = np.zeros((len(rows), 7))
    dummies[np.arange(len(rows)), dayofweek[7:]] = 1.0
    x = np.hstack([*blocks, dummies])
    y = prices[7:, :]
    return x, y


def forecast_day(
    prices: np.ndarray, exog: np.ndarray, dayofweek: np.ndarray
) -> np.ndarray:
    """Fit on all complete days and predict the last day (whose price row is unused).

    Inputs cover the calibration window plus the target day as the final row;
    prices[-1] may be NaN. Returns the 24 predicted prices.
    """
    x_all, y_all = build_xy(prices, exog, dayofweek)
    x_train, y_train = x_all[:-1], y_all[:-1]
    x_pred = x_all[-1:]

    n_dummies = 7
    scaler_x = InvariantScaler().fit(x_train[:, :-n_dummies])
    scaler_y = InvariantScaler().fit(y_train)  # column-wise: one median/mad per hour

    xs_train = np.hstack([scaler_x.transform(x_train[:, :-n_dummies]), x_train[:, -n_dummies:]])
    xs_pred = np.hstack([scaler_x.transform(x_pred[:, :-n_dummies]), x_pred[:, -n_dummies:]])
    ys_train = scaler_y.transform(y_train)

    out = np.empty((1, 24))
    n, p = xs_train.shape
    for h in range(24):
        # modern sklearn requires an explicit noise variance when n <= p (the
        # 56/84-day windows); the paper's old sklearn estimated it implicitly,
        # so short-window results may deviate slightly from the published ones
        kwargs = {"noise_variance": float(np.var(ys_train[:, h]))} if n <= p + 1 else {}
        selector = LassoLarsIC(criterion="aic", max_iter=2500, **kwargs)
        selector.fit(xs_train, ys_train[:, h])
        model = Lasso(alpha=selector.alpha_, max_iter=2500)
        model.fit(xs_train, ys_train[:, h])
        out[0, h] = model.predict(xs_pred)[0]
    return scaler_y.inverse_transform(out)[0]


def rolling_forecast(
    df: pd.DataFrame,
    price_col: str,
    exog_cols: list[str],
    test_start: pd.Timestamp,
    calibration_window: int,
    progress_every: int = 50,
) -> pd.Series:
    """Daily-recalibrated LEAR forecasts for every day from test_start to the end.

    `df` is hourly with exactly 24 rows per day. Exogenous values of the target
    day are used (day-ahead forecasts: known pre-auction).
    """
    daily_index = pd.DatetimeIndex(sorted({t.normalize() for t in df.index}))
    test_days = daily_index[daily_index >= test_start.normalize()]

    prices_all = df[price_col].to_numpy().reshape(-1, 24)
    exog_all = np.stack([df[c].to_numpy().reshape(-1, 24) for c in exog_cols], axis=2)
    dow_all = np.array([d.dayofweek for d in daily_index])

    preds = []
    for i, day in enumerate(test_days):
        d = daily_index.get_loc(day)
        lo = max(0, d - calibration_window)
        sl = slice(lo, d + 1)
        preds.append(forecast_day(prices_all[sl], exog_all[sl], dow_all[sl]))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  day {i + 1}/{len(test_days)}", flush=True)

    hours = df.index[df.index >= test_days[0]]
    return pd.Series(np.concatenate(preds), index=hours[: len(preds) * 24], name="lear_forecast")
