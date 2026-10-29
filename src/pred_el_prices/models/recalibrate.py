"""Rolling PIT recalibration of percentile forecasts.

A forecast's 99 percentiles define a predictive CDF F. If the model is
calibrated, u = F(actual) is uniform. Recalibration learns where the actual
prices really fall (the empirical CDF G of past u values) and serves the
percentile level tau from the model's level G^-1(tau): if the model's "90th"
was exceeded 13 % of the time, tau = 0.9 is taken from a higher model level.

Leakage: for the week starting S, G is fitted on forecast hours of days up to
S-2 (their actuals are known at the gate of S's first day), within a trailing
window. Beyond the 1st/99th percentiles the model CDF is extended with an
exponential tail fitted to the 95th-99th (5th-1st) percentile spacing, so levels
outside [0.01, 0.99] exist when recalibration asks for them.
"""

import numpy as np
import pandas as pd

LEVELS = np.arange(1, 100) / 100.0
_LN5 = np.log(5.0)  # (1 - 0.95) / (1 - 0.99)
# the production rule (nets and t0 alike): PIT of forecast days <= D-2 in a trailing
# 365-day window; with fewer PIT days the raw percentiles pass through, flagged
PIT_WINDOW_DAYS = 365
MIN_PIT_DAYS = 28


def pit(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """F(y) per row; q (n, 99) sorted percentiles, y (n,). Exponential tails outside."""
    n = len(y)
    k = (q < y[:, None]).sum(axis=1)  # number of percentiles below y
    u = np.empty(n)
    inner = (k > 0) & (k < 99)
    i = np.where(inner)[0]
    lo, hi = q[i, k[i] - 1], q[i, k[i]]
    frac = np.where(hi > lo, (y[i] - lo) / np.where(hi > lo, hi - lo, 1.0), 0.5)
    u[i] = LEVELS[k[i] - 1] + frac * 0.01
    up = k == 99
    s_up = np.maximum(q[up, 98] - q[up, 94], 1e-6) / _LN5
    u[up] = 1.0 - 0.01 * np.exp(-(y[up] - q[up, 98]) / s_up)
    dn = k == 0
    s_dn = np.maximum(q[dn, 4] - q[dn, 0], 1e-6) / _LN5
    u[dn] = 0.01 * np.exp(-(q[dn, 0] - y[dn]) / s_dn)
    return u


def quantile_at(q: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Model quantiles at arbitrary levels taus (m,), for every row: (n, m)."""
    out = np.empty((len(q), len(taus)))
    for j, t in enumerate(taus):
        if t > 0.99:
            s = np.maximum(q[:, 98] - q[:, 94], 1e-6) / _LN5
            out[:, j] = q[:, 98] + s * np.log(0.01 / (1.0 - t))
        elif t < 0.01:
            s = np.maximum(q[:, 4] - q[:, 0], 1e-6) / _LN5
            out[:, j] = q[:, 0] - s * np.log(0.01 / t)
        else:
            pos = (t - 0.01) * 100.0
            a = min(int(np.floor(pos)), 97)
            w = pos - a
            out[:, j] = (1 - w) * q[:, a] + w * q[:, a + 1]
    return out


def recalibrate_day(
    raw: np.ndarray,
    history: pd.DataFrame,
    actual: pd.Series,
    delivery: pd.Timestamp,
    window_days: int = PIT_WINDOW_DAYS,
    min_days: int = MIN_PIT_DAYS,
    bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, int]:
    """Rolling PIT recalibration of one delivery day's percentiles `raw` (24, 99).

    `history`: past hourly forecasts, 99 sorted percentile columns (raw, as the model
    made them); only days <= D-2 inside the window count (their prices are known at D's
    gate). Returns (percentiles, number of PIT days used); with fewer than `min_days` the
    raw percentiles pass through unchanged and 0 is returned. `bounds`: clip the
    recalibrated percentiles (the SDAC limits).
    """
    last = delivery - pd.Timedelta(days=2)
    h = history.dropna()
    days = h.index.normalize()
    a = actual.reindex(h.index)
    sel = (days <= last) & (days > last - pd.Timedelta(days=window_days)) & a.notna().to_numpy()
    n_days = int(days[sel].nunique())
    if n_days < min_days:
        return raw, 0
    u = pit(h[sel].to_numpy(dtype=float), a[sel].to_numpy())
    taus = np.clip(np.quantile(u, LEVELS), 1e-4, 1 - 1e-4)
    q = np.sort(quantile_at(raw, taus), axis=1)
    return (np.clip(q, *bounds) if bounds is not None else q), n_days


def recalibrate(
    qdf: pd.DataFrame,
    q_cols: list[str],
    first: str,
    window_days: int = 365,
    step: str = "7D",
) -> pd.DataFrame:
    """Recalibrated copy of qdf (q_cols + actual), from `first` on; earlier rows dropped."""
    q_all = qdf[q_cols].to_numpy()
    u_all = pit(q_all, qdf["actual"].to_numpy())
    days = qdf.index.normalize()
    starts = pd.date_range(pd.Timestamp(first, tz="UTC"), qdf.index.max(), freq=step)
    parts = []
    for s in starts:
        e = s + pd.Timedelta(step)
        last = s - pd.Timedelta(days=2)
        fit = (days <= last) & (days > last - pd.Timedelta(days=window_days))
        test = (qdf.index >= s) & (qdf.index < e)
        if not test.any() or not fit.any():
            continue
        taus = np.quantile(u_all[fit], LEVELS)  # G^-1 at each nominal level
        new = np.sort(quantile_at(q_all[test], np.clip(taus, 1e-4, 1 - 1e-4)), axis=1)
        parts.append(pd.DataFrame(new, index=qdf.index[test], columns=q_cols))
    out = pd.concat(parts)
    out["actual"] = qdf["actual"].reindex(out.index)
    return out
