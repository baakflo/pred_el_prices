"""Production networks: weekly ensemble training and the daily gate-time forecast.

Weekly (`train_ensemble`): 12 JSU + 12 quantile-head networks (seeds 0..11 per head),
tuned body, fuel scaling, neighbour LOAD, trained on the 1460 UTC days ending on
monday - 2, exactly as the weekly backtests and the production replay
(`experiments/qnn_replay.py`) train. One bundle file per ensemble (models.qnn.save_bundle).

Daily (`forecast_day`), for delivery block D (UTC day, the site convention), from what
exists at the gate (12:00 CET/CEST on D-1), mirroring the replay's gate-safe row:
- German wind + solar for D: our own RES forecast, never the TSO's (18:00 D-1).
- DE and neighbour load for D: TSO day-ahead forecasts, UTC 22-23 filled from 24 h
  earlier (they belong to the next local day, unpublished at the gate).
- Lags d-1 (UTC 00-21 only, gate-safe design), d-2, d-3, d-7 from the dataset.
- Fuels: settlements lagged 2 days, as features/dataset.py builds them.
Then: Vincentize (mean of all 24 networks' percentiles, sorted), clip to the SDAC
limits, recalibrate with the rolling 365-day PIT window (forecast days <= D-2), clip
again; quarter-hours = hourly percentiles + one median shape (models/qh_shape.py),
the shape model refit on history through D-2.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pred_el_prices.experiments.qnn_de import (
    NEIGHBOUR_COLS,
    PRICE_BOUNDS,
    PRICE_COL,
    Q_COLS,
    RES_COLS,
    days_from_frame,
    design,
    fuel_cost,
)
from pred_el_prices.features.dataset import FUEL_SETTLEMENT_LAG_DAYS, NEIGHBOUR_ZONES
from pred_el_prices.models import qh_shape
from pred_el_prices.models.qnn import FittedNet, QNNConfig, fit, predict
from pred_el_prices.models.recalibrate import LEVELS, pit, quantile_at
from pred_el_prices.pipeline import cache
from pred_el_prices.pipeline.entsoe import resample_hourly
from pred_el_prices.production.site import QH_START, R_COLS

HEADS = ("jsu", "quantile")
N_SEEDS = 12
WINDOW_DAYS = 1460
TRAIN_START = "2018-12-01"
N_UNSCALED = 7  # day-of-week dummies
# tuned body (Optuna, qnn-de-20260923-182551); the replay's defaults
BODY = {
    "hidden": [256, 256, 256],
    "dropout": 0.2784723989915054,
    "lr": 0.0004983021010966105,
    "weight_decay": 8.548006178626775e-05,
    "batch_size": 32,
}
PIT_WINDOW_DAYS = 365
MIN_PIT_DAYS = 28  # fewer days of PIT history: publish uncalibrated, flagged
LOAD_NB_COLS = [c.replace("rl_", "load_") for c in NEIGHBOUR_COLS]


# ---------------------------------------------------------------- weekly training


def monday_of(day: pd.Timestamp) -> pd.Timestamp:
    """The Monday starting the week that contains `day` (the backtests' refit weeks)."""
    return day.normalize() - pd.Timedelta(days=day.dayofweek)


def next_monday(day: pd.Timestamp) -> pd.Timestamp:
    """`day` if it is a Monday, else the following Monday (the weekly job's target week)."""
    return day.normalize() + pd.Timedelta(days=(7 - day.dayofweek) % 7)


def _fit_one(x, y, config, seed):
    import torch

    torch.set_num_threads(1)
    return fit(x, y, N_UNSCALED, config, seed)


def training_set(dataset: pd.DataFrame, monday: pd.Timestamp, window_days: int = WINDOW_DAYS):
    """X, Y and row days for the fit serving the week of `monday`: days <= monday - 2."""
    missing = [
        c
        for c in [PRICE_COL, "load_forecast_mw", *RES_COLS, *LOAD_NB_COLS, "ttf_gas_eur_mwh"]
        if c not in dataset.columns
    ]
    if missing:
        raise RuntimeError(f"dataset lacks network inputs {missing}; seed the caches first")
    last = monday - pd.Timedelta(days=2)
    # nothing after the last training day may shorten the contiguous stretch
    ds = dataset[dataset.index < last + pd.Timedelta(days=1)]
    prices, exog, fuels, days, _ = days_from_frame(ds, TRAIN_START, [], neighbours="load")
    x, y, row_days = design(prices / fuel_cost(fuels)[:, None], exog, fuels, days)
    train = (row_days <= last) & (row_days > last - pd.Timedelta(days=window_days))
    return x[train], y[train], row_days[train]


def train_ensemble(
    dataset: pd.DataFrame,
    monday: pd.Timestamp,
    n_seeds: int = N_SEEDS,
    heads: tuple[str, ...] = HEADS,
    n_jobs: int = -1,
    window_days: int = WINDOW_DAYS,
    min_days: int = 1000,
    config_overrides: dict | None = None,
) -> tuple[list[FittedNet], dict]:
    """Fit the weekly ensemble; returns (networks, meta) for models.qnn.save_bundle."""
    from joblib import Parallel, delayed

    x, y, row_days = training_set(dataset, monday, window_days)
    if len(row_days) < min_days:
        raise RuntimeError(
            f"only {len(row_days)} contiguous training days before {monday:%Y-%m-%d} "
            f"(need {min_days}): a gap in the cache cut the window"
        )
    last = monday - pd.Timedelta(days=2)
    if row_days.max() != last:
        print(f"WARN: training ends {row_days.max():%Y-%m-%d}, not {last:%Y-%m-%d} (stale cache)")
    configs = {h: QNNConfig(**{**BODY, **(config_overrides or {})}, head=h) for h in heads}
    jobs = [(h, s) for h in heads for s in range(n_seeds)]
    nets = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_fit_one)(x, y, configs[h], s) for h, s in jobs
    )
    meta = {
        "version": 1,
        "monday": f"{monday:%Y-%m-%d}",
        "trained_through": f"{row_days.max():%Y-%m-%d}",
        "train_first": f"{row_days.min():%Y-%m-%d}",
        "n_train_days": len(row_days),
        "window_days": window_days,
        "heads": [h for h, _ in jobs],
        "seeds": [s for _, s in jobs],
        "fuel_scale": True,
        "neighbours": "load",
        "n_features": int(x.shape[1]),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return nets, meta


# ------------------------------------------------------------ the gate-time row


def gate_inputs(
    cache_dir: Path, delivery: pd.Timestamp, load_de_fallback: pd.Series | None = None
) -> dict:
    """Load forecasts (DE, summed neighbours) and fuels for `delivery`, as at its gate.

    UTC 22-23 of the delivery block are always taken from 24 h earlier, whether or not
    the cache has them by now (a backfill must see what the gate saw). Other missing
    hours: up to 2 filled from 24 h earlier; more on DE falls back to
    `load_de_fallback` (the surrogate LEAR used, flagged); more on a neighbour fails.
    """
    hours = pd.date_range(delivery - pd.Timedelta(days=1), periods=48, freq="1h", tz="UTC")
    day = hours[24:]
    flags = {"load_surrogate": False}

    def at_gate(series: pd.Series, name: str) -> pd.Series | None:
        s = series.reindex(hours).copy()
        s.iloc[46:48] = s.iloc[22:24].to_numpy()
        gap = s.iloc[24:].isna()
        if gap.sum() > 2:
            return None
        if gap.any():
            print(f"nets: {name} load, {int(gap.sum())} hour(s) filled from 24 h earlier")
            s.iloc[24:] = s.iloc[24:].fillna(pd.Series(s.iloc[:24].to_numpy(), index=day))
        return None if s.iloc[24:].isna().any() else s.iloc[24:]

    de = cache.load(cache_dir, "entsoe/load_forecast")
    load_de = at_gate(resample_hourly(de[["Forecasted Load"]])["Forecasted Load"], "DE")
    if load_de is None:
        if load_de_fallback is None:
            raise RuntimeError(f"DE load forecast for {delivery:%Y-%m-%d} missing")
        load_de = load_de_fallback.reindex(day)
        flags["load_surrogate"] = True
    nb = pd.Series(0.0, index=day)
    for zone in NEIGHBOUR_ZONES:
        z = cache.load(cache_dir, f"entsoe/{zone}/load_forecast")
        s = (
            at_gate(resample_hourly(z[["Forecasted Load"]])["Forecasted Load"], zone)
            if len(z)
            else None
        )
        if s is None:
            raise RuntimeError(f"{zone} load forecast for {delivery:%Y-%m-%d} missing")
        nb += s

    fuels = cache.load(cache_dir, "fuels_daily")
    lagged = fuels.copy()
    lagged.index = lagged.index + pd.Timedelta(days=FUEL_SETTLEMENT_LAG_DAYS)
    # as the dataset: the settlement row at-or-before D (a NaN in it stays NaN there);
    # then as load_days: TTF carried forward from the last valid value, EUA NaN -> 0
    at = lagged.reindex([delivery], method="ffill")
    ttf = at["ttf_gas_eur_mwh"].iloc[0]
    if pd.isna(ttf):
        ttf = lagged["ttf_gas_eur_mwh"].loc[:delivery].dropna().iloc[-1]
    eua = at["eua_proxy_usd"].iloc[0] if "eua_proxy_usd" in at else np.nan
    fuel = np.array([ttf, 0.0 if pd.isna(eua) else eua])
    return {"load_de": load_de, "load_nb": nb, "fuel": fuel, "flags": flags}


def gate_row(
    dataset: pd.DataFrame,
    delivery: pd.Timestamp,
    load_de: pd.Series,
    load_nb: pd.Series,
    res: pd.Series,
    fuel: np.ndarray,
) -> tuple[np.ndarray, float]:
    """The network input row for `delivery` (1, n_features) and its fuel-cost scale.

    History (D-7..D-1) comes from `dataset` exactly as training builds it
    (experiments.qnn_de.load_days, neighbours="load"); the delivery day's exogenous
    block from the arguments. D-1's UTC 22-23 may be absent: the gate-safe design drops
    the lag-1 hours 22-23 on every row, so they never reach the network.
    """
    days = pd.date_range(delivery - pd.Timedelta(days=7), periods=8, freq="D")
    hist = pd.date_range(days[0], periods=7 * 24, freq="1h")
    ds = dataset.reindex(hist)
    res_hist = ds[RES_COLS].sum(axis=1, min_count=len(RES_COLS))
    nb_hist = dataset[LOAD_NB_COLS].ffill().reindex(hist).sum(axis=1, min_count=len(LOAD_NB_COLS))
    ttf = dataset["ttf_gas_eur_mwh"].ffill().reindex(hist)
    eua = ds["eua_proxy_usd"].fillna(0.0) if "eua_proxy_usd" in ds else pd.Series(0.0, hist)

    prices = np.vstack([ds[PRICE_COL].to_numpy().reshape(7, 24), np.full((1, 24), np.nan)])
    exog = np.stack(
        [
            np.concatenate([ds["load_forecast_mw"].to_numpy(), load_de.to_numpy()]),
            np.concatenate([res_hist.to_numpy(), res.to_numpy()]),
            np.concatenate([nb_hist.to_numpy(), load_nb.to_numpy()]),
        ],
        axis=1,
    ).reshape(8, 24, 3)
    # settlements are daily (constant within a UTC day), so the day's first hour equals
    # training's 24-hour mean; the absent D-1 22-23 rows must not enter it
    fuels = np.vstack([np.column_stack([ttf, eua]).reshape(7, 24, 2)[:, 0, :], fuel[None, :]])
    scale = fuel_cost(fuels)
    x, _, _ = design(prices / scale[:, None], exog, fuels, days)
    if np.isnan(x).any():
        raise RuntimeError(f"incomplete network inputs for {delivery:%Y-%m-%d}")
    return x, float(scale[-1])


# ------------------------------------------------------------- predict + calibrate


def predict_raw(nets: list[FittedNet], x: np.ndarray, scale: float) -> np.ndarray:
    """Vincentized percentiles (24, 99) EUR/MWh: mean over networks, sorted, clipped."""
    q = np.mean([predict(n, x)[0] for n in nets], axis=0) * scale
    return np.clip(np.sort(q, axis=-1), *PRICE_BOUNDS)


def recalibrate_day(
    raw: np.ndarray,
    history: pd.DataFrame,
    actual: pd.Series,
    delivery: pd.Timestamp,
    window_days: int = PIT_WINDOW_DAYS,
    min_days: int = MIN_PIT_DAYS,
) -> tuple[np.ndarray, int]:
    """Rolling PIT recalibration (models.recalibrate, one day at a time).

    `history`: raw percentiles (R_COLS) of past forecast hours; only days <= D-2 inside
    the window count (their prices are known at D's gate). Returns (percentiles, number
    of PIT days used); with fewer than `min_days` the raw percentiles pass through.
    """
    last = delivery - pd.Timedelta(days=2)
    h = history[R_COLS].dropna()
    days = h.index.normalize()
    a = actual.reindex(h.index)
    sel = (days <= last) & (days > last - pd.Timedelta(days=window_days)) & a.notna().to_numpy()
    n_days = int(days[sel].nunique())
    if n_days < min_days:
        return raw, 0
    u = pit(h[sel].to_numpy(dtype=float), a[sel].to_numpy())
    taus = np.clip(np.quantile(u, LEVELS), 1e-4, 1 - 1e-4)
    q = np.sort(quantile_at(raw, taus), axis=1)
    return np.clip(q, *PRICE_BOUNDS), n_days


def load_history(seed_path: Path | None, log_path: Path | None) -> pd.DataFrame:
    """Past hourly forecasts: raw percentiles (R_COLS) and the published median (q50).

    The seed (backtest/replay quantiles, `seed_pit`) covers the time before go-live; the
    nets log wins wherever both exist.
    """
    from pred_el_prices.production.site import read_log

    parts = []
    if seed_path is not None and Path(seed_path).exists():
        parts.append(pd.read_parquet(seed_path)[[*R_COLS, "q50"]])
    if log_path is not None and Path(log_path).exists():
        log = read_log(log_path)
        parts.append(log.loc[log["kind"] == "h", [*R_COLS, "q50"]])
    if not parts:
        return pd.DataFrame(columns=[*R_COLS, "q50"], index=pd.DatetimeIndex([], tz="UTC"))
    hist = pd.concat(parts)
    return hist[~hist.index.duplicated(keep="last")].sort_index()


# ------------------------------------------------------------------ quarter-hours


@dataclass
class QHInputs:
    """Quarter-hour shape inputs from the caches: TSO 15-min load forecast, TSO hourly
    wind/solar (training side only), 15-min prices (targets)."""

    load15: pd.Series
    solar_h: pd.Series
    wind_h: pd.Series
    price15: pd.Series

    @classmethod
    def from_cache(cls, cache_dir: Path) -> "QHInputs":
        from pred_el_prices.production.site import quarter_prices

        start = QH_START - pd.Timedelta(days=2)
        load15 = cache.load(cache_dir, "entsoe/load_forecast", start=start)["Forecasted Load"]
        ws = resample_hourly(cache.load(cache_dir, "entsoe/wind_solar_forecast", start=start))
        return cls(
            load15.dropna(),
            ws["Solar"],
            ws["Wind Onshore"] + ws["Wind Offshore"],
            quarter_prices(cache_dir),
        )


def quarter_forecast(
    q_hourly: pd.DataFrame,
    delivery: pd.Timestamp,
    res_parts: pd.DataFrame,
    p_hist: pd.Series,
    qh: QHInputs,
) -> tuple[pd.DataFrame, bool]:
    """96 quarter-hour percentile rows; (frame, shaped). Without the delivery day's
    15-min load forecast the hourly percentiles are repeated x4 (shaped=False)."""
    quarters = pd.date_range(delivery, periods=96, freq="15min")
    p_hour = pd.concat([p_hist[p_hist.index < delivery].dropna(), q_hourly["q50"]])

    # gate-time inputs for the delivery day (as replay_page_data.py): the ramps need the
    # hour before and after; UTC 22-23 load from 24 h earlier; own RES held flat outside
    hrs = pd.date_range(delivery - pd.Timedelta(hours=1), periods=26, freq="1h")
    ld = qh.load15.reindex(pd.date_range(hrs[0], periods=104, freq="15min"))
    b = ld.index >= delivery + pd.Timedelta(hours=22)
    ld[b] = qh.load15.reindex(ld.index[b] - pd.Timedelta(days=1)).to_numpy()
    solar = res_parts["solar_forecast_mw"].reindex(hrs).ffill().bfill()
    wind = (
        (res_parts["wind_onshore_forecast_mw"] + res_parts["wind_offshore_forecast_mw"])
        .reindex(hrs)
        .ffill()
        .bfill()
    )
    if ld.reindex(quarters).isna().any():
        print(f"nets: no 15-min load forecast for {delivery:%Y-%m-%d}; hourly repeated x4")
        return qh_shape.quarter_percentiles(q_hourly, pd.Series(0.0, index=quarters), Q_COLS), False
    f_d = qh_shape.features(ld, solar, wind, p_hour).reindex(quarters)

    y = qh_shape.target(qh.price15)
    f_all = qh_shape.features(qh.load15, qh.solar_h, qh.wind_h, p_hour)
    tr = f_all.index[f_all.index.normalize() <= delivery - pd.Timedelta(days=2)]
    f_tr = f_all.loc[tr.intersection(y.index)].dropna()
    if len(f_tr) < 96 * MIN_PIT_DAYS:
        print(f"nets: {len(f_tr)} shape training quarters (no median history?); repeated x4")
        return qh_shape.quarter_percentiles(q_hourly, pd.Series(0.0, index=quarters), Q_COLS), False
    shape = qh_shape.fit_predict(f_tr, y, f_d)
    q = qh_shape.quarter_percentiles(q_hourly, shape, Q_COLS)
    q[Q_COLS] = np.clip(q[Q_COLS].to_numpy(), *PRICE_BOUNDS)
    return q, True


# ---------------------------------------------------------------------- one day


def forecast_day(
    nets: list[FittedNet],
    meta: dict,
    delivery: pd.Timestamp,
    dataset: pd.DataFrame,
    cache_dir: Path,
    res_parts: pd.DataFrame,
    history: pd.DataFrame,
    prices: pd.Series,
    qh: QHInputs,
    load_de_fallback: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Hourly (Q_COLS final + R_COLS raw) and quarter-hour (Q_COLS) percentiles, and flags.

    `res_parts`: own RES forecast per target for the delivery day (daily_forecast.
    own_res_parts); `history`: load_history(); `prices`: hourly clearing prices (PIT).
    """
    from pred_el_prices.daily_forecast import own_res_total

    trained = pd.Timestamp(meta["trained_through"], tz="UTC")
    if trained > delivery - pd.Timedelta(days=2):
        raise RuntimeError(
            f"bundle trained through {meta['trained_through']}: too late for delivery "
            f"{delivery:%Y-%m-%d} (its targets were not known at the gate)"
        )
    inputs = gate_inputs(cache_dir, delivery, load_de_fallback)
    res = own_res_total(res_parts)
    x, scale = gate_row(
        dataset, delivery, inputs["load_de"], inputs["load_nb"], res, inputs["fuel"]
    )
    raw = predict_raw(nets, x, scale)
    q, n_days = recalibrate_day(raw, history, prices, delivery)
    hours = pd.date_range(delivery, periods=24, freq="1h")
    hourly = pd.DataFrame(np.hstack([q, raw]), index=hours, columns=Q_COLS + R_COLS)
    quarters, shaped = quarter_forecast(hourly[Q_COLS], delivery, res_parts, history["q50"], qh)
    flags = {
        "trained_through": meta["trained_through"],
        "recal_days": n_days,
        "shaped": shaped,
        **inputs["flags"],
    }
    return hourly, quarters, flags


# ------------------------------------------------------------------ PIT seeding


def seed_pit(
    quantiles_path: Path,
    cal_path: Path | None,
    before: pd.Timestamp,
    logs: tuple[Path, ...] = (),
) -> pd.DataFrame:
    """PIT/shape history for launch: raw percentiles of a backtest run over the 400 days
    before `before` (R_COLS) and its recalibrated median (q50, for the shape model's
    training; the 15-minute history starts 2025-10-01, inside that span), plus
    the hourly rows of any nets logs (e.g. a production-path backfill), which win."""
    from pred_el_prices.production.site import read_log

    raw = pd.read_parquet(quantiles_path)
    # the PIT window is 365 days; a margin covers the weekly-stale start of production
    keep = (raw.index < before) & (raw.index >= before - pd.Timedelta(days=PIT_WINDOW_DAYS + 35))
    raw = raw.loc[keep, Q_COLS].set_axis(R_COLS, axis=1)
    raw["q50"] = (
        pd.read_parquet(cal_path)["q50"].reindex(raw.index) if cal_path is not None else np.nan
    )
    parts = [raw]
    for p in logs:
        log = read_log(Path(p))
        parts.append(log.loc[log["kind"] == "h", [*R_COLS, "q50"]])
    seed = pd.concat(parts)
    seed = seed[~seed.index.duplicated(keep="last")].sort_index().astype("float32")
    seed.index.name = "t"
    return seed
