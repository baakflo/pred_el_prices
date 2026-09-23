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
from joblib import Parallel, delayed
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


def hinge_features(rl: np.ndarray, quantiles: tuple[float, float]) -> np.ndarray:
    """Piecewise residual-load terms, (n_days, 48): max(0, k_lo - RL) and max(0, RL - k_hi).

    rl: (n_days, 24) residual load; the last row is the target day. Knots are
    quantiles of RL over the other rows (the calibration window only), and
    both terms are divided by that window's RL MAD so they enter on a
    comparable scale — they bypass the asinh scaling, which would turn a
    mostly-zero column (MAD 0 -> 1) into a log of megawatts.
    """
    window = rl[:-1]
    k_lo, k_hi = np.quantile(window, quantiles)
    mad = np.median(np.abs(window - np.median(window))) / 0.6745
    return np.hstack([np.maximum(0.0, k_lo - rl), np.maximum(0.0, rl - k_hi)]) / mad


def forecast_day(
    prices: np.ndarray,
    exog: np.ndarray,
    dayofweek: np.ndarray,
    hinge_quantiles: tuple[float, float] | None = None,
) -> np.ndarray:
    """Fit on all complete days and predict the last day (whose price row is unused).

    Inputs cover the calibration window plus the target day as the final row;
    prices[-1] may be NaN. Returns the 24 predicted prices.

    `hinge_quantiles` adds same-day (lag 0) hinge terms on residual load
    exog[..., 0] - exog[..., 1] (i.e. exog must be [load, res]); see
    hinge_features. None reproduces plain LEAR exactly.
    """
    x_all, y_all = build_xy(prices, exog, dayofweek)
    n_unscaled = 7  # day-of-week dummies
    if hinge_quantiles is not None:
        hinges = hinge_features(exog[:, :, 0] - exog[:, :, 1], hinge_quantiles)
        x_all = np.hstack([x_all, hinges[7:]])
        n_unscaled += hinges.shape[1]
    x_train, y_train = x_all[:-1], y_all[:-1]
    x_pred = x_all[-1:]

    scaler_x = InvariantScaler().fit(x_train[:, :-n_unscaled])
    scaler_y = InvariantScaler().fit(y_train)  # column-wise: one median/mad per hour

    xs_train = np.hstack([scaler_x.transform(x_train[:, :-n_unscaled]), x_train[:, -n_unscaled:]])
    xs_pred = np.hstack([scaler_x.transform(x_pred[:, :-n_unscaled]), x_pred[:, -n_unscaled:]])
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
    n_jobs: int = 1,
    predict_exog: pd.DataFrame | None = None,
    hinge_quantiles: tuple[float, float] | None = None,
    gate_safe_prices: bool = False,
) -> pd.Series:
    """Daily-recalibrated LEAR forecasts for every day from test_start to the end.

    `df` is hourly with exactly 24 rows per day. Exogenous values of the target
    day are used (day-ahead forecasts: known pre-auction). Every day's
    recalibration is independent, so n_jobs parallelizes over test days with
    identical results to the serial run (the fit is deterministic).

    `predict_exog` (hourly, columns a subset of exog_cols) replaces the exog of
    the PREDICTION day only — calibration rows keep the published values, which
    is production's information set when the official series appears post-gate
    and a substitute must be used pre-gate. Days without a complete 24-hour
    override keep the published values.

    `hinge_quantiles` is passed to forecast_day (knots re-estimated per window).

    `gate_safe_prices`: UTC hours 22-23 of the day before the target belong to
    the target's LOCAL delivery day, i.e. to the very auction being forecast.
    Plain LEAR sees them as lag-1 prices; production cannot, and heals them
    from 24h-lag (daily_forecast.lear_forecast). True mirrors that healing.
    """
    daily_index = pd.DatetimeIndex(sorted({t.normalize() for t in df.index}))
    test_days = daily_index[daily_index >= test_start.normalize()]

    prices_all = df[price_col].to_numpy().reshape(-1, 24)
    exog_all = np.stack([df[c].to_numpy().reshape(-1, 24) for c in exog_cols], axis=2)
    dow_all = np.array([d.dayofweek for d in daily_index])

    override: dict[pd.Timestamp, np.ndarray] = {}
    override_cols: list[int] = []
    if predict_exog is not None:
        cols = [c for c in exog_cols if c in predict_exog.columns]
        override_cols = [exog_cols.index(c) for c in cols]
        clean = predict_exog[cols].dropna()
        for day, chunk in clean.groupby(clean.index.normalize()):
            if len(chunk) == 24:
                override[day] = chunk.to_numpy()

    def _one_day(day: pd.Timestamp) -> np.ndarray:
        d = daily_index.get_loc(day)
        lo = max(0, d - calibration_window)
        sl = slice(lo, d + 1)
        exog_window = exog_all[sl]
        if day in override:
            exog_window = exog_window.copy()
            exog_window[-1, :, override_cols] = override[day].T
        prices_window = prices_all[sl]
        if gate_safe_prices:
            prices_window = prices_window.copy()
            prices_window[-2, 22:] = prices_window[-3, 22:]
        return forecast_day(prices_window, exog_window, dow_all[sl], hinge_quantiles)

    if n_jobs == 1:
        preds = []
        for i, day in enumerate(test_days):
            preds.append(_one_day(day))
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  day {i + 1}/{len(test_days)}", flush=True)
    else:
        preds = Parallel(n_jobs=n_jobs, verbose=5 if progress_every else 0)(
            delayed(_one_day)(day) for day in test_days
        )

    hours = df.index[df.index >= test_days[0]]
    return pd.Series(np.concatenate(preds), index=hours[: len(preds) * 24], name="lear_forecast")
