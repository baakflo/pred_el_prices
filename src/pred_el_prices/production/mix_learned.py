"""Learned nets+t0 mixes M1 and M3, shadow-logged only (pandas/numpy/scipy, no torch).

The equal-weight mix (production/mix.py) with a fitted nets weight instead of 0.5, level
by level on the 99-level grid: q = w * q_nets + (1 - w) * q_t0 (clipped to the SDAC
limits), both inputs exactly as the equal mix has them (nets final = recalibrated, t0
recalibrated by T0Calibration, on the grid by t0_on_grid). The variants and settings are
the neural repo's learned-mix experiment (`_scratch/mix/`, runs/mix-learned, plan Step 1,
results 2026-11-03), chosen there on 2025-10..2026-03:

- **M1**: one w per UTC delivery hour, refit every run. Minimises, over the fit window,
  sum_h [L_h(w_h) + LAMBDA (w_h - 0.5)^2] + MU sum_h (w_h - w_{h+1})^2, L_h = the window
  mean of the hour's mean 99-level pinball; exact on a 0.02 grid of w by dynamic
  programming along the hour chain (UTC 23 -> 00 not wrapped).
- **M3**: w = sigmoid(theta . x), x = intercept + 16 gate-safe features standardised on
  the window rows (clipped to +-5 sd); minimises the window's mean 99-level pinball +
  RIDGE |theta|^2 (theta = 0 <=> w = 0.5) by L-BFGS-B, refit every run (the experiment
  refit weekly, warm-started; here cold from theta = 0 daily, under a second).

Fit window: the last WINDOW_DAYS calendar days up to D-2 (gate-safe, as the PIT
recalibration), days with both models and all 24 clearing prices. Fewer than MIN_DAYS such
days (M3: with complete features): w = 0.5 (the equal mix), flagged `fallback`.

History (`learned_history`): per day, nets = the standing nets log rows (days from
NETS_LIVE_FROM on), else the seed's; t0 = the standing t0 log rows recalibrated (each day
on its own PIT history, as the site does), else the seed's; M3 features = those logged
with M3's rows, else the seed's, else rebuilt from the dataset (`proxy`, proxy_features).
The seed (`mix_learned_seed.parquet`, `pep seed-mix-learned`) is the experiment's inputs:
the nets replay (morning) and the recalibrated t0-B backtest.

Logged every run with t0 (both editions, the nets run's flags, evening included) to
`mix_m1_log/` and `mix_m3_log/` (monthly partitions, the mix log's layout: 24 "h" + 96
"qh" rows, quarter-hours = the nets' shifted by what the mix moved their hour), plus
`variant`, `w` (the hour's weight, on its quarters too), `fallback`, `fit_days`,
`seed_days`; M3's hourly rows also carry its raw features (FEAT_COLS) and `proxy_days`.
Never published, never a primary; any failure costs these rows only (::warning::).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pred_el_prices.models.recalibrate import (
    LEVELS,
    MIN_PIT_DAYS,
    PIT_WINDOW_DAYS,
    pit,
    quantile_at,
)
from pred_el_prices.production.mix import (
    CARRY,
    PRICE_BOUNDS,
    WITH_T0,
    T0Calibration,
    _evening,
    mix_quarters,
    read_t0,
    t0_on_grid,
)
from pred_el_prices.production.site import (
    LOG_DIR,
    Q_COLS,
    _write_partitions,
    log_exists,
    read_log,
    read_partitions,
)
from pred_el_prices.production.t0_shadow import Q_NAMES as T0_COLS

VARIANTS = ("m1", "m3")
LOG_DIRS = {"m1": "mix_m1_log", "m3": "mix_m3_log"}
SEED_NAME = "mix_learned_seed.parquet"
WINDOW_DAYS = 90
MIN_DAYS = 60
SEED_DAYS = 180  # seed span before the first live day (the window is 90)
# the live nets before this day ran with EUA = 0 (ragged yfinance rows, fixed 2026-10-15,
# nets judged from 10-17 on): their rows would pull w toward t0, so the fit skips them
NETS_LIVE_FROM = pd.Timestamp("2026-10-17", tz="UTC")
GRID = np.linspace(0.0, 1.0, 51)  # w on a 0.02 grid
M1_LAMBDA, M1_MU = 1.0, 3.0
M3_RIDGE, M3_MAXITER = 0.1, 300
N_COLS = [f"n{c[1:]}" for c in Q_COLS]  # seed: nets final percentiles
T_COLS = [f"t{c[1:]}" for c in Q_COLS]  # seed: t0 recalibrated, on the 99 grid
# raw M3 inputs per hour: DE load (the run's input: TSO day-ahead forecast, UTC 22-23
# from 24 h earlier, or the surrogate), own RES of D (MW), holiday share of the
# Berlin-local day, mean clearing price of D-1 UTC 00-21
FEAT_COLS = ["f_load", "f_res", "f_holiday", "f_price_dm1"]
M3_FEATURES = [
    "sin1", "cos1", "sin2", "cos2", "sin3", "cos3", "sat", "sun", "holiday",
    "resid_load", "res_share", "price_dm1", "spread", "abs_spread", "width_nets",
    "width_t0",
]  # fmt: skip
RES_COLS = ["wind_onshore_forecast_mw", "wind_offshore_forecast_mw", "solar_forecast_mw"]
I10, I50, I90 = 9, 49, 89
DAY = pd.Timedelta(days=1)


def hours_of(day: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(day, periods=24, freq="1h")


# ------------------------------------------------------------------ the mixes


def weighted(qn: np.ndarray, qt: np.ndarray, w) -> np.ndarray:
    """Level-by-level mix, w broadcastable to qn (the nets' weight), clipped."""
    return np.clip(qt + w * (qn - qt), *PRICE_BOUNDS)


def _pinball(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    e = y[..., None] - q
    return np.maximum(LEVELS * e, (LEVELS - 1) * e).mean(axis=-1)


def m1_losses(qn: np.ndarray, qt: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(24, len(GRID)): window mean of each hour's mean 99-level pinball at w = GRID[k].
    qn, qt (days, 24, 99), y (days, 24)."""
    out = np.empty((qn.shape[1], len(GRID)))
    for k, w in enumerate(GRID):
        out[:, k] = _pinball(weighted(qn, qt, w), y).mean(axis=0)
    return out


def chain_dp(unary: np.ndarray, pair: np.ndarray) -> np.ndarray:
    """argmin_k sum_h unary[h, k_h] + sum_h pair[k_h, k_h+1] (exact); indices (hours,)."""
    n, k = unary.shape
    cost = unary[0].copy()
    back = np.zeros((n, k), dtype=np.int64)
    for h in range(1, n):
        tot = cost[:, None] + pair
        back[h] = tot.argmin(axis=0)
        cost = tot.min(axis=0) + unary[h]
    ks = np.empty(n, dtype=np.int64)
    ks[-1] = cost.argmin()
    for h in range(n - 1, 0, -1):
        ks[h - 1] = back[h, ks[h]]
    return ks


def fit_m1(qn, qt, y, lam: float = M1_LAMBDA, mu: float = M1_MU) -> np.ndarray:
    """M1's 24 weights (on GRID) from the window's days."""
    unary = m1_losses(qn, qt, y) + lam * (GRID - 0.5) ** 2
    return GRID[chain_dp(unary, mu * (GRID[:, None] - GRID[None, :]) ** 2)]


def m3_raw(qn: np.ndarray, qt: np.ndarray, feat: np.ndarray, days: pd.DatetimeIndex):
    """(days, 24, 16) M3 features from both models' percentiles and FEAT_COLS (days, 24, 4)."""
    n = len(days)
    h = np.broadcast_to(np.arange(24.0)[None, :], (n, 24))
    wd = np.broadcast_to(np.asarray(days.dayofweek)[:, None], (n, 24))
    load, res, hol, dm1 = (feat[..., j] for j in range(4))
    cols = []
    for k in (1, 2, 3):
        cols += [np.sin(2 * np.pi * k * h / 24), np.cos(2 * np.pi * k * h / 24)]
    spread = qn[..., I50] - qt[..., I50]
    cols += [
        (wd == 5).astype(float), (wd == 6).astype(float), hol,
        (load - res) / 1000.0, res / load, dm1,
        spread, np.abs(spread),
        qn[..., I90] - qn[..., I10], qt[..., I90] - qt[..., I10],
    ]  # fmt: skip
    return np.stack(cols, axis=-1).astype(np.float64)


def _design(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    z = np.clip((x - mu) / sd, -5, 5)
    return np.concatenate([np.ones((len(z), 1)), z], axis=1)


def _m3_objective(theta, X, qt, dq, y, ridge):
    w = 1.0 / (1.0 + np.exp(-(X @ theta)))
    e = y[:, None] - (qt + w[:, None] * dq)
    pb = np.maximum(LEVELS * e, (LEVELS - 1) * e)
    g = LEVELS - (e < 0)
    dj_dw = (-(g * dq)).mean(axis=1)
    grad = X.T @ (dj_dw * w * (1 - w)) / len(y) + 2 * ridge * theta
    return pb.mean() + ridge * theta @ theta, grad


@dataclass
class M3Fit:
    theta: np.ndarray
    mu: np.ndarray
    sd: np.ndarray

    def weights(self, x: np.ndarray) -> np.ndarray:
        """w per row of raw features x (rows, 16)."""
        return 1.0 / (1.0 + np.exp(-(_design(x, self.mu, self.sd) @ self.theta)))


def fit_m3(x, qn, qt, y, ridge: float = M3_RIDGE, theta0=None) -> M3Fit:
    """M3 from the window's days: x (days, 24, 16) raw features, qn/qt (days, 24, 99)."""
    from scipy.optimize import minimize

    xr = x.reshape(-1, x.shape[-1])
    mu, sd = xr.mean(axis=0), xr.std(axis=0) + 1e-9
    X = _design(xr, mu, sd)
    qt_r = qt.reshape(-1, qt.shape[-1]).astype(np.float64)
    dq = qn.reshape(-1, qn.shape[-1]).astype(np.float64) - qt_r
    theta0 = np.zeros(X.shape[1]) if theta0 is None else theta0
    opt = minimize(
        _m3_objective, theta0, args=(X, qt_r, dq, y.reshape(-1), ridge), jac=True,
        method="L-BFGS-B", options={"maxiter": M3_MAXITER},
    )  # fmt: skip
    return M3Fit(opt.x, mu, sd)


# ------------------------------------------------------------------ history


@dataclass
class History:
    """Per day (sorted): both models' percentiles (days, 24, 99), prices (days, 24), raw M3
    features (days, 24, 4; NaN where unknown), and where each day came from."""

    days: pd.DatetimeIndex
    qn: np.ndarray
    qt: np.ndarray
    y: np.ndarray
    feat: np.ndarray
    seed: np.ndarray  # bool: a component from the seed
    proxy: np.ndarray  # bool: features rebuilt from the dataset

    def window(self, delivery: pd.Timestamp, need_features: bool = False) -> np.ndarray:
        """Indices of the fit days for `delivery`: the WINDOW_DAYS calendar days up to D-2,
        complete days only (with finite features too when `need_features`)."""
        last = delivery - 2 * DAY
        ok = (self.days <= last) & (self.days > last - WINDOW_DAYS * DAY)
        ok &= np.isfinite(self.qn).all(axis=(1, 2)) & np.isfinite(self.qt).all(axis=(1, 2))
        ok &= np.isfinite(self.y).all(axis=1)
        if need_features:
            ok &= np.isfinite(self.feat).all(axis=(1, 2)) & (self.feat[..., 0] != 0).all(axis=1)
        return np.where(ok)[0]


def _complete_days(frame: pd.DataFrame | None, cols, lo, hi) -> dict:
    """{day: 24 x cols array} for the days in [lo, hi] with 24 finite rows."""
    if frame is None or frame.empty:
        return {}
    f = frame[(frame.index >= lo) & (frame.index < hi + DAY)]
    out = {}
    for day, part in f.groupby(f.index.normalize()):
        part = part[~part.index.duplicated(keep="last")].reindex(hours_of(day))
        v = part[cols].to_numpy(dtype=float)
        if np.isfinite(v).all():
            out[day] = v
    return out


def recalibrate_days(cal: T0Calibration, t0_days: dict) -> dict:
    """{day: t0 recalibrated (24, 99)} for raw t0 day frames, each day on its own PIT history:
    recalibrate_day's rule (cal.hourly), with the history's PIT computed once."""
    hist = cal.history.dropna()
    a = cal.prices.reindex(hist.index)
    ok = a.notna().to_numpy()
    hist_days = hist.index.normalize()[ok]
    u = pit(hist.to_numpy(dtype=float)[ok], a.to_numpy()[ok]) if ok.any() else np.empty(0)
    out = {}
    for day, t0 in t0_days.items():
        g = t0_on_grid(t0.sort_index()).to_numpy()
        last = day - 2 * DAY
        sel = (hist_days <= last) & (hist_days > last - PIT_WINDOW_DAYS * DAY)
        if hist_days[sel].nunique() < MIN_PIT_DAYS:
            out[day] = g
            continue
        taus = np.clip(np.quantile(u[sel], LEVELS), 1e-4, 1 - 1e-4)
        out[day] = np.clip(np.sort(quantile_at(g, taus), axis=1), *PRICE_BOUNDS)
    return out


def day_features(
    day: pd.Timestamp, prices: pd.Series, load: pd.Series, res: pd.Series
) -> np.ndarray:
    """(24, 4) FEAT_COLS of one day from its load and RES inputs (indexed by its hours)."""
    from pred_el_prices.features.holidays import holiday_share

    hrs = hours_of(day)
    dm1 = prices.reindex(pd.date_range(day - DAY, periods=22, freq="1h")).mean()
    return np.stack(
        [
            load.reindex(hrs).to_numpy(dtype=float),
            res.reindex(hrs).to_numpy(dtype=float),
            holiday_share(hrs, 0).to_numpy(dtype=float),
            np.full(24, dm1, dtype=float),
        ],
        axis=-1,
    )


def proxy_features(day: pd.Timestamp, prices: pd.Series, dataset: pd.DataFrame) -> np.ndarray:
    """FEAT_COLS of a past live day the M3 log lacks: the TSO load forecast (UTC 22-23
    from 24 h earlier where missing, as the run's input), and the TSO RES forecast in
    place of the own RES the run had (not logged; for a day <= D-2 a published value)."""
    hrs = hours_of(day)
    load_fc = dataset["load_forecast_mw"]
    load = load_fc.reindex(hrs)
    load = load.fillna(pd.Series(load_fc.reindex(hrs - DAY).to_numpy(), index=hrs))
    res = dataset[RES_COLS].sum(axis=1, min_count=len(RES_COLS))
    return day_features(day, prices, load, res)


def learned_history(
    out_dir: Path,
    delivery: pd.Timestamp,
    prices: pd.Series,
    cal: T0Calibration,
    dataset: pd.DataFrame | None = None,
) -> History:
    """Fit history for `delivery`: the window's days (<= D-2) from the seed and the live
    logs in `out_dir`, live components winning (module docstring)."""
    hi = delivery - 2 * DAY
    lo = hi - (WINDOW_DAYS - 1) * DAY
    seed_path = Path(out_dir) / SEED_NAME
    seed = pd.read_parquet(seed_path) if seed_path.exists() else None
    s_n = _complete_days(seed, N_COLS, lo, hi)
    s_t = _complete_days(seed, T_COLS, lo, hi)
    s_f = _complete_days(seed, FEAT_COLS, lo, hi) if seed is not None else {}
    s_y = seed["y"] if seed is not None and "y" in seed else None

    path = Path(out_dir) / LOG_DIR
    nets = read_log(path) if log_exists(path) else None
    if nets is not None:
        nets = nets[(nets["kind"] == "h") & (nets.index >= NETS_LIVE_FROM)]
    l_n = _complete_days(nets, Q_COLS, lo, hi)
    t0_log = read_t0(out_dir)
    l_t_raw = {}
    if t0_log is not None:
        t0_log = t0_log[(t0_log.index >= lo) & (t0_log.index < hi + DAY)]
        for day, part in t0_log.groupby(t0_log.index.normalize()):
            if len(part) == 24 and np.isfinite(part[T0_COLS].to_numpy(dtype=float)).all():
                l_t_raw[day] = part
    l_t = recalibrate_days(cal, l_t_raw)
    m3 = read_partitions(Path(out_dir) / LOG_DIRS["m3"])
    if m3 is not None and set(FEAT_COLS) <= set(m3.columns):
        l_f = _complete_days(m3[m3["kind"] == "h"], FEAT_COLS, lo, hi)
    else:
        l_f = {}

    days = sorted({*l_n, *s_n} & {*l_t, *s_t})
    n = len(days)
    qn, qt = np.full((n, 24, 99), np.nan), np.full((n, 24, 99), np.nan)
    y, feat = np.full((n, 24), np.nan), np.full((n, 24, 4), np.nan)
    from_seed, proxy = np.zeros(n, bool), np.zeros(n, bool)
    for i, day in enumerate(days):
        hrs = hours_of(day)
        qn[i] = l_n[day] if day in l_n else s_n[day]
        qt[i] = l_t[day] if day in l_t else s_t[day]
        from_seed[i] = day not in l_n or day not in l_t
        a = prices.reindex(hrs).to_numpy(dtype=float)
        if s_y is not None and np.isnan(a).any():
            a = np.where(np.isnan(a), s_y.reindex(hrs).to_numpy(dtype=float), a)
        y[i] = a
        if day in l_f:
            feat[i] = l_f[day]
        elif day in s_f:
            feat[i] = s_f[day]
        elif dataset is not None:
            feat[i] = proxy_features(day, prices, dataset)
            proxy[i] = True
    days = pd.DatetimeIndex(days).tz_convert("UTC") if days else pd.DatetimeIndex([], tz="UTC")
    return History(days, qn, qt, y, feat, from_seed, proxy)


# ------------------------------------------------------------------ one run


def learned_weights(
    variant: str,
    hist: History,
    delivery: pd.Timestamp,
    qn_d: np.ndarray,
    qt_d: np.ndarray,
    feat_d: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """(24 weights, info) of one variant for `delivery` (qn_d, qt_d: its (24, 99) inputs;
    feat_d: its (24, 4) FEAT_COLS, M3 only). Below MIN_DAYS fit days: 0.5, `fallback`."""
    idx = hist.window(delivery, need_features=variant == "m3")
    info = {
        "fit_days": len(idx),
        "seed_days": int(hist.seed[idx].sum()),
        "fallback": len(idx) < MIN_DAYS,
    }
    if variant == "m3":
        info["proxy_days"] = int(hist.proxy[idx].sum())
        if feat_d is None or not np.isfinite(feat_d).all():
            raise RuntimeError("M3 features of the delivery day incomplete")
    if info["fallback"]:
        return np.full(24, 0.5), info
    qn, qt, y = hist.qn[idx], hist.qt[idx], hist.y[idx]
    if variant == "m1":
        return fit_m1(qn, qt, y), info
    if variant == "m3":
        fit = fit_m3(m3_raw(qn, qt, hist.feat[idx], hist.days[idx]), qn, qt, y)
        x_d = m3_raw(qn_d[None], qt_d[None], feat_d[None], pd.DatetimeIndex([delivery]))[0]
        return fit.weights(x_d), info
    raise ValueError(f"unknown variant {variant!r}")


def learned_rows(
    variant: str,
    nets_rows: pd.DataFrame,
    t0_99: pd.DataFrame,
    w: np.ndarray,
    info: dict,
    generated_utc: str,
    recal_days: int,
    feat_d: np.ndarray | None = None,
) -> pd.DataFrame:
    """One run's log rows of one variant (the mix log's layout, see the module docstring)."""
    h = nets_rows[nets_rows["kind"] == "h"].sort_index()
    qh = nets_rows[nets_rows["kind"] == "qh"].sort_index()
    if len(h) != 24 or len(qh) != 96:
        raise RuntimeError(f"nets rows incomplete ({len(h)} hours, {len(qh)} quarters)")
    qt = t0_99.reindex(h.index)[Q_COLS].to_numpy(dtype=float)
    if np.isnan(qt).any():
        raise RuntimeError("t0 rows do not cover the nets' hours")
    mix_h = pd.DataFrame(
        weighted(h[Q_COLS].to_numpy(dtype=float), qt, w[:, None]), index=h.index, columns=Q_COLS
    )
    mix_qh = mix_quarters(qh, h, mix_h)
    values = pd.concat([mix_h.astype("float32"), mix_qh.astype("float32")])
    meta = pd.DataFrame(
        {
            "kind": ["h"] * 24 + ["qh"] * 96,
            "generated_utc": generated_utc,
            "nets_generated_utc": str(h["generated_utc"].iloc[0]),
            "components": WITH_T0,
            "variant": variant,
            "w": np.concatenate([w, np.repeat(w, 4)]).astype("float32"),
        },
        index=values.index,
    )
    for col in CARRY:
        if col in h:
            meta[col] = h[col].iloc[0]
    meta["t0_recal_days"] = recal_days
    for key, value in info.items():
        meta[key] = value
    if feat_d is not None:
        for j, col in enumerate(FEAT_COLS):
            meta[col] = np.concatenate([feat_d[:, j], np.full(96, np.nan)])
    rows = pd.concat([meta, values], axis=1)
    rows.index.name = "t"
    return rows


def learned_step(
    out_dir: Path,
    delivery: pd.Timestamp,
    t0: pd.DataFrame | None,
    generated_utc: str,
    evening: bool = False,
    prices: pd.Series | None = None,
    load_d: pd.Series | None = None,
    own_res: pd.Series | None = None,
    dataset: pd.DataFrame | None = None,
) -> dict[str, bool]:
    """M1 and M3 for `delivery` from its standing nets rows (this run's edition, as
    mix_step) and this run's t0 rows, appended to mix_m1_log/ and mix_m3_log/. Never
    raises: a failure logs a ::warning:: and skips that variant's rows (or both).

    `prices`: hourly clearing prices (required: the fit and t0's recalibration). `load_d`,
    `own_res`: the run's DE load and own RES inputs for D (M3's features); `dataset`: the
    run's dataset (M3 features of past days the M3 log lacks). Returns {variant: logged}.
    """
    done = dict.fromkeys(VARIANTS, False)
    try:
        if t0 is None:
            print(f"mix learned: no t0 rows for {delivery:%Y-%m-%d}, nothing logged")
            return done
        if prices is None:
            raise RuntimeError("clearing prices required")
        path = Path(out_dir) / LOG_DIR
        log = read_log(path) if log_exists(path) else None
        rows = log[log.index.normalize() == delivery] if log is not None else None
        if rows is None or rows.empty or _evening(rows[rows["kind"] == "h"]) != evening:
            print(
                f"mix learned: no {'evening' if evening else 'morning'} nets for {delivery:%Y-%m-%d}"
            )
            return done
        cal = T0Calibration.from_dir(out_dir, prices)
        t0_99, recal = cal.hourly(t0)
        h = rows[rows["kind"] == "h"].sort_index()
        qn_d = h[Q_COLS].to_numpy(dtype=float)
        qt_d = t0_99.reindex(h.index)[Q_COLS].to_numpy(dtype=float)
        if len(h) != 24 or np.isnan(qt_d).any():
            raise RuntimeError("t0 rows do not cover the nets' hours")
        hist = learned_history(out_dir, delivery, prices, cal, dataset)
    except Exception as e:  # noqa: BLE001 - shadow path: nothing here may cost the day
        print(f"::warning::learned mix step failed ({type(e).__name__}: {e})")
        return done
    for variant in VARIANTS:
        try:
            feat_d = None
            if variant == "m3":
                if load_d is None or own_res is None:
                    raise RuntimeError("the run's load / own RES inputs are missing")
                feat_d = day_features(delivery, prices, load_d, own_res)
            w, info = learned_weights(variant, hist, delivery, qn_d, qt_d, feat_d)
            out = learned_rows(variant, rows, t0_99, w, info, generated_utc, recal, feat_d)
            _write_partitions(Path(out_dir) / LOG_DIRS[variant], out)
            fb = ", FALLBACK w = 0.5" if info["fallback"] else ""
            print(
                f"mix learned {variant}: {delivery:%Y-%m-%d} logged ({info['fit_days']} fit "
                f"days, {info['seed_days']} from the seed, mean w {w.mean():.3f}{fb})"
            )
            done[variant] = True
        except Exception as e:  # noqa: BLE001 - one variant's rows, never the day
            print(f"::warning::learned mix {variant} failed ({type(e).__name__}: {e})")
    return done


# ------------------------------------------------------------------ seed


def seed_mix_learned(
    nets_logs: tuple[Path, ...],
    t0_quantiles: Path,
    inputs_dir: Path,
    prices: pd.Series,
    before: pd.Timestamp,
    days: int = SEED_DAYS,
) -> pd.DataFrame:
    """The learned mixes' seed: hourly rows for the `days` days before `before` (the first
    live day), with the nets' final percentiles (N_COLS), t0 recalibrated on the 99 grid
    (T_COLS), the clearing price `y` and the raw M3 features (FEAT_COLS).

    As the experiment built them (`_scratch/mix/prep.py`): nets = the morning replay log
    (`runs/replay-2526/morning/nets_log.parquet`; several logs: the later wins), t0 = the
    t0-B backtest (`runs/t0-compare/t0_B.parquet`) by PCHIP onto the grid, recalibrated by
    the nets' rule with t0-B's own rows as PIT history; features from the t0 comparison's
    inputs (`runs/t0-compare/inputs/`: hourly.parquet load_tso / holiday_share / price,
    own_res.parquet = the morning own RES per delivery day). A day needs both models;
    features may be missing (M3 then skips the day).
    """
    from pred_el_prices.production.site import _standing

    parts = []
    for p in nets_logs:
        log = pd.read_parquet(p)
        log.index = pd.to_datetime(log.index, utc=True)
        parts.append(log[log["kind"] == "h"])
    nets = _standing(pd.concat(parts))[Q_COLS].astype(float)
    t0b = pd.read_parquet(t0_quantiles)
    t0b.index = pd.to_datetime(t0b.index, utc=True)
    t0b = t0b[T0_COLS].dropna()
    t0b = t0b[~t0b.index.duplicated(keep="last")].sort_index()
    cal = T0Calibration(t0b, prices)
    hourly = pd.read_parquet(Path(inputs_dir) / "hourly.parquet")
    hourly.index = pd.to_datetime(hourly.index, utc=True)
    own = pd.read_parquet(Path(inputs_dir) / "own_res.parquet")
    own.index = pd.to_datetime(own.index, utc=True)
    own = own[own.index.normalize() == pd.to_datetime(own["delivery_day"], utc=True)]
    own = own[~own.index.duplicated(keep="last")]["own_res"]

    lo, hi = before - days * DAY, before - DAY
    n_days = _complete_days(nets, Q_COLS, lo, hi)
    t_raw = {}
    for day in n_days:
        part = t0b.reindex(hours_of(day))
        if np.isfinite(part.to_numpy(dtype=float)).all():
            t_raw[day] = part
    t_days = recalibrate_days(cal, t_raw)
    out = []
    for day in sorted(t_days):
        hrs = hours_of(day)
        f = pd.DataFrame(np.concatenate([n_days[day], t_days[day]], axis=1), hrs, N_COLS + T_COLS)
        f = f.astype("float32")
        f["y"] = prices.reindex(hrs).to_numpy(dtype=float)
        dm1 = hourly["price"].reindex(pd.date_range(day - DAY, periods=22, freq="1h")).mean()
        f["f_load"] = hourly["load_tso"].reindex(hrs).to_numpy(dtype=float)
        f["f_res"] = own.reindex(hrs).to_numpy(dtype=float)
        f["f_holiday"] = hourly["holiday_share"].reindex(hrs).to_numpy(dtype=float)
        f["f_price_dm1"] = dm1
        out.append(f)
    if not out:
        raise RuntimeError("no day with both models in the seed span")
    seed = pd.concat(out)
    seed.index.name = "t"
    return seed
