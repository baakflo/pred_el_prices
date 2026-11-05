"""Shadow forecast of t0-beta fine-tuned on 13 covariate rows ("ALL"): logged, never published.

The Step-4 winner of the t0 fine-tuning study (pred_el_prices_nl plan, 2026-11-05: report
half M0 mix 4.905 vs 5.090 zero-shot over three seeds). A forward test next to the
zero-shot shadow (production/t0_shadow.py): same origin, context, horizon and quantile
levels, rows to `t0_all_log/YYYY-MM.parquet`, in no site JSON, fail-soft.

The weights are a full state dict on top of the pinned base revision (`load_model`);
`predict()` runs unchanged (the fine-tuning forward reproduced it bit for bit). Rows, in
the order the checkpoint was trained on (it carries the list; a mismatch refuses to run):
- rl, hol: as the zero-shot shadow (t0_shadow.build_inputs);
- load: context TSO load; horizon as rl's load (the run's own DE load on the UTC hours of
  D, else the TSO forecast of local D, local D+1 from 24 h earlier);
- wind_on, wind_off, solar: context the TSO forecast of each part; horizon the own RES
  forecast of each part (t0_shadow.horizon_res, part by part);
- ttf, eua: last settlement <= D-2 (NaN and zero rows dropped before the forward fill,
  the EUA=0 guard); horizon flat at the value in force at D 00 UTC;
- sat, sun, mon, bridge: Berlin-local calendar flags (bridge = working day between a
  federal holiday and a weekend);
- nb_sum: summed TSO load forecasts of the 9 neighbour zones, horizon by the training rule
  (local D from the forecast, other hours from 24 h earlier).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from pred_el_prices.daily_forecast import RES_TARGETS, own_res_parts, own_res_total
from pred_el_prices.features.dataset import FUEL_SETTLEMENT_LAG_DAYS, NEIGHBOUR_ZONES
from pred_el_prices.features.holidays import holiday_share
from pred_el_prices.features.tso_areas import entsoe_de_series
from pred_el_prices.pipeline import cache
from pred_el_prices.pipeline.entsoe import resample_hourly
from pred_el_prices.production import t0_shadow as T0
from pred_el_prices.production.site import _write_partitions

ROWS = ["rl", "hol", "load", "wind_on", "wind_off", "solar", "ttf", "eua",
        "sat", "sun", "mon", "bridge", "nb_sum"]  # fmt: skip
PARTS = dict(zip(["wind_on", "wind_off", "solar"], RES_TARGETS, strict=True))
FUELS = {"ttf": "ttf_gas_eur_mwh", "eua": "eua_proxy_usd"}
LOG_DIR = "t0_all_log"
DEFAULT_CHECKPOINT = Path("models/t0_all.pt")
H, DAY = T0.H, T0.DAY


def calendar_flags(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """sat/sun/mon/bridge of each hour's Berlin-local day. Bridge day: a working day
    (Mon-Fri, no federal holiday) between a federal holiday and a weekend."""
    local = idx.tz_convert("Europe/Berlin")
    days = pd.DatetimeIndex(sorted(set(local.date)))
    span = pd.date_range(days[0] - DAY, days[-1] + DAY)
    noon = span.tz_localize("Europe/Berlin") + pd.Timedelta(hours=12)
    fed = pd.Series(holiday_share(noon, 0).to_numpy() >= 1.0, index=span.date)
    wkend = pd.Series(span.dayofweek >= 5, index=span.date)
    off = fed | wkend
    prev_off, next_off = off.shift(1, fill_value=False), off.shift(-1, fill_value=False)
    prev_fed, next_fed = fed.shift(1, fill_value=False), fed.shift(-1, fill_value=False)
    prev_we, next_we = wkend.shift(1, fill_value=False), wkend.shift(-1, fill_value=False)
    bridge = ~off & prev_off & next_off & ((prev_fed & next_we) | (prev_we & next_fed))
    dow = local.dayofweek
    return pd.DataFrame(
        {
            "sat": (dow == 5).astype(float),
            "sun": (dow == 6).astype(float),
            "mon": (dow == 0).astype(float),
            "bridge": bridge.reindex(pd.Index(local.date)).to_numpy().astype(float),
        },
        index=idx,
    )


def gate_safe_fuels(fuels_daily: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """TTF and EUA at each hour: the last settlement <= D-2, NaN and zero rows dropped
    before the forward fill (a ragged or zero newest row must not enter)."""
    out = pd.DataFrame(index=idx)
    for name, col in FUELS.items():
        s = fuels_daily[col].dropna() if col in fuels_daily else pd.Series(dtype=float)
        s = s[s != 0]
        s.index = s.index + pd.Timedelta(days=FUEL_SETTLEMENT_LAG_DAYS)
        out[name] = s.reindex(idx, method="ffill") if len(s) else np.nan
    return out


def lagged_horizon(series: pd.Series, hz: pd.DatetimeIndex, day: pd.Timestamp) -> np.ndarray:
    """Training's horizon rule for a TSO load series: local day D from the series, other
    hours from 24 h earlier; remaining gaps from the same hour 1 or 2 days earlier."""
    src = hz.where(hz.tz_convert("Europe/Berlin").date == day.date(), hz - DAY)
    v = series.reindex(src).to_numpy(dtype=float, copy=True)
    for k in (1, 2):
        gap = np.isnan(v)
        if gap.any():
            v[gap] = series.reindex(src[gap] - k * DAY).to_numpy()
    return v


def gate_loads(cache_dir: Path, start: pd.Timestamp) -> dict[str, pd.Series]:
    """The pre-gate TSO load forecasts straight from the cache: the DE aggregate (areas
    stepped in) and the 9-neighbour sum. `dataset` rows exist only where a price cleared,
    so at the morning run it lacks local day D; training read D's forecast (its days
    had cleared), so the horizon must too."""
    zones = []
    for z in NEIGHBOUR_ZONES:
        df = cache.load(cache_dir, f"entsoe/{z}/load_forecast", start=start)
        if len(df):
            zones.append(resample_hourly(df[["Forecasted Load"]])["Forecasted Load"].rename(z))
    nb = (
        pd.concat(zones, axis=1).sum(axis=1, min_count=len(NEIGHBOUR_ZONES))
        if len(zones) == len(NEIGHBOUR_ZONES)
        else pd.Series(dtype=float)
    )
    return {"load": entsoe_de_series(cache_dir, "load_forecast_mw", start=start), "nb": nb}


def build_inputs(delivery, prices, dataset, fuels_daily, load_d, res_d, res_prev,
                 context=T0.CONTEXT, gate=None):  # fmt: skip
    """Model inputs for `delivery`: (context prices, covariates [13, context + HORIZON],
    kept horizon hours, keep mask). `res_d` / `res_prev`: own RES parts (columns
    RES_TARGETS) of D and of D-1 (or None). `gate`: `gate_loads`, filling the horizon
    hours `dataset` lacks (the context stays the dataset's, as in training). Raises on a
    missing horizon value."""
    o = T0.origin_of(delivery)
    ctx = pd.date_range(o - context * H, periods=context, freq="1h")
    hz = pd.date_range(o, periods=T0.HORIZON, freq="1h")
    keep = hz.normalize() == delivery
    every = ctx.append(hz)
    ctx_prices = prices.reindex(ctx).to_numpy(np.float32)
    nb_cols = [f"load_{z.lower()}_mw" for z in NEIGHBOUR_ZONES]
    nb = dataset[nb_cols].sum(axis=1, min_count=len(nb_cols))
    load_src, nb_src = dataset["load_forecast_mw"], nb
    if gate is not None:
        load_src = load_src.combine_first(gate["load"])
        nb_src = nb_src.combine_first(gate["nb"])

    load_c = dataset["load_forecast_mw"].reindex(ctx).to_numpy(dtype=float)
    load_h = T0.horizon_load(load_src, load_d, hz, delivery)
    parts = {}
    for row, col in PARTS.items():
        prev = None if res_prev is None else res_prev[col]
        parts[row] = (
            dataset[col].reindex(ctx).to_numpy(dtype=float),
            T0.horizon_res(res_d[col], prev, hz, delivery),
        )
    res_c = sum(c for c, _ in parts.values())
    # the zero-shot shadow's total: own RES total over the horizon, D-1 boundary hours too
    res_h = T0.horizon_res(
        own_res_total(res_d), None if res_prev is None else own_res_total(res_prev), hz, delivery
    )
    fuel = gate_safe_fuels(fuels_daily, every)
    cal = calendar_flags(every)

    rows = {
        "rl": (load_c - res_c, load_h - res_h),
        "hol": (holiday_share(every, 0).to_numpy() >= 1.0).astype(float),
        "load": (load_c, load_h),
        **parts,
        **{
            f: (fuel[f].reindex(ctx).to_numpy(), np.full(len(hz), fuel[f].get(delivery, np.nan)))
            for f in FUELS
        },
        **{c: cal[c].to_numpy() for c in ["sat", "sun", "mon", "bridge"]},
        "nb_sum": (nb.reindex(ctx).to_numpy(dtype=float), lagged_horizon(nb_src, hz, delivery)),
    }
    cov = []
    for r in ROWS:
        v = rows[r]
        v = np.concatenate(v) if isinstance(v, tuple) else v
        if np.isnan(v[context:]).any():
            raise RuntimeError(f"row {r}: horizon incomplete for {delivery:%Y-%m-%d}")
        cov.append(v)
    return ctx_prices, np.stack(cov).astype(np.float32), hz[keep], keep


def load_model(checkpoint: Path):
    """The base revision with the fine-tuned state dict on top (strict: every tensor)."""
    import torch

    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ck.get("rows") != ROWS:
        raise RuntimeError(f"checkpoint rows {ck.get('rows')} != {ROWS}")
    if ck.get("model_revision") != T0.MODEL_REVISION:
        raise RuntimeError(f"checkpoint base {ck.get('model_revision')} != {T0.MODEL_REVISION}")
    model = T0.load_model()
    model.load_state_dict(ck["state"], strict=True)
    return model.eval(), str(ck.get("name", checkpoint.name))


def forecast_day(model, delivery, prices, dataset, fuels_daily, load_d, res_d, res_prev,
                 gate=None):  # fmt: skip
    ctx, cov, hours, keep = build_inputs(
        delivery, prices, dataset, fuels_daily, load_d, res_d, res_prev, gate=gate
    )
    fc = model.predict(
        ctx[None], horizon=T0.HORIZON, quantile_levels=T0.LEVELS, future_covariates=cov[None]
    )
    q = fc.quantiles.float().cpu().numpy()[0]
    out = pd.DataFrame(q[keep], index=hours, columns=T0.Q_NAMES)
    out["median"] = out["q50"]
    return out


def t0_all_step(
    out_dir: Path,
    cache_dir: Path,
    dataset: pd.DataFrame,
    features: pd.DataFrame,
    delivery: pd.Timestamp,
    prices: pd.Series,
    load_d: pd.Series,
    res_parts: pd.DataFrame,
    generated_utc: str,
    weather_vintage: str,
    load_surrogate: bool,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    model=None,
) -> bool:
    """Shadow forecast of the fine-tuned t0 for `delivery`, appended to its log. Never
    raises; skips (False) when the checkpoint is absent. `model`: (model, name), tests."""
    try:
        if model is None and not Path(checkpoint).exists():
            print(f"t0-all shadow: no checkpoint at {checkpoint}, skipped")
            return False
        t = time.perf_counter()
        prev = delivery - DAY
        try:  # own RES parts of D-1 as the D-2 morning run made them (as t0_shadow)
            seen = dataset[dataset.index < prev - 2 * H]
            res_prev = own_res_parts(features, seen, cache_dir, prev)
        except Exception as e:  # noqa: BLE001 - a fallback exists for these two hours
            print(f"t0-all: own RES of {prev:%Y-%m-%d} unavailable ({e}); D's + 24 h used")
            res_prev = None
        model, name = load_model(Path(checkpoint)) if model is None else model
        fuels_daily = cache.load(cache_dir, "fuels_daily")
        gate = gate_loads(cache_dir, delivery - 7 * DAY)
        rows = forecast_day(
            model, delivery, prices, dataset, fuels_daily, load_d, res_parts, res_prev, gate
        ).astype("float32")
        rows.insert(0, "generated_utc", generated_utc)
        rows.insert(1, "weather_vintage", weather_vintage)
        rows.insert(2, "load_surrogate", bool(load_surrogate))
        rows.insert(3, "checkpoint", name)
        rows.index.name = "t"
        _write_partitions(Path(out_dir) / LOG_DIR, rows)
        print(f"t0-all shadow: {delivery:%Y-%m-%d} logged in {time.perf_counter() - t:.0f} s")
        return True
    except Exception as e:  # noqa: BLE001 - shadow path: nothing here may cost LEAR the day
        print(f"::warning::t0-all shadow step failed ({type(e).__name__}: {e})")
        return False
