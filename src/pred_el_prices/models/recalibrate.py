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
