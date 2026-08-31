"""Daily pre-gate production forecast for the public site.

Runs on D-1 before the 12:00 CET/CEST auction gate: refreshes the input
caches, archives today's 00Z ECMWF ENS run (falling back, when allowed, to
yesterday's 12Z run pre-archived by the evening cron — a stale-but-present
weather vintage beats a missed day), generates the own pre-gate RES
forecast for the next UTC day (no public wind/solar forecast exists
pre-gate — see the 2026-08-15 plan addendum), feeds it to LEAR(364,
academic exog) via the predict-day-only substitution, and emits the site
JSON contract (latest.json, history.json) plus an append-only forecast log.

Delivery days are UTC blocks, matching the backtest convention. The load
forecast for the delivery day comes from ENTSO-E (published >= 2 h before
gate closure by law); if it is not yet available the run aborts loudly so a
later cron slot can retry.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from pred_el_prices.features.dataset import build_dataset
from pred_el_prices.features.ens_weather import run_features
from pred_el_prices.models.lear import forecast_day
from pred_el_prices.pipeline import cache
from pred_el_prices.pipeline.capacity import hourly_capacity
from pred_el_prices.pipeline.entsoe import resample_hourly

MODEL_LABEL = "LEAR(364, academic exog) + own-RES v2"
NOTE_GATE_OK = "Generated before the 12:00 CET/CEST auction gate."
NOTE_GATE_MISSED = (
    "Generated AFTER the 12:00 CET/CEST auction gate closed — flagged for "
    "honesty; not comparable to pre-gate days."
)
NOTE_RES = {
    "00Z": (
        "Renewables input is our own forecast from the ECMWF ENS 00Z run (the "
        "official TSO forecast is only published at 18:00, after the auction)."
    ),
    "12Z": (
        "Renewables input is our own forecast from the previous evening's ECMWF "
        "ENS 12Z run — the morning 00Z run was unavailable (measured fallback "
        "cost: ~+0.1 EUR/MWh MAE)."
    ),
}
NOTE_LOAD_SURROGATE = (
    "Load input is a surrogate model (weather + calendar + last week's load) — "
    "the ENTSO-E load forecast was unavailable (measured cost: ~+0.3 EUR/MWh MAE)."
)
NOTE_EVENING = (
    "Evening edition, built the night before from the ECMWF ENS 12Z run and a "
    "load surrogate (measured cost: ~+0.4 EUR/MWh MAE vs the regular morning "
    "forecast, which replaces this one when its inputs publish)."
)
RES_TARGETS = {
    "wind_onshore_forecast_mw": "wind_onshore_capacity_mw",
    "wind_offshore_forecast_mw": "wind_offshore_capacity_mw",
    "solar_forecast_mw": "solar_capacity_mw",
}
WINDOW = 364
LAG_BURN_IN_DAYS = 7


def update_features(
    features_path: Path,
    archive_dir: Path,
    through_run_date,
    allow_fallback: bool = False,
) -> pd.DataFrame:
    """Append hourly ENS features for delivery days missing from the table.

    Bookkeeping is per delivery day: the 00Z run of D-1 is the primary
    vintage; when its archive file is absent and `allow_fallback` is set,
    the 12Z run of D-2 (on S3 since the previous evening) fills the day in.
    Fallback rows are replaced by primary rows once the 00Z file gets
    backfilled, so the table converges to the vintage the backtests used.
    """
    features = pd.read_parquet(features_path)

    def archive_path(run_day, run_hour: int) -> Path:
        stem = f"ecmwf-ens_{run_day:%Y%m%d}{run_hour:02d}.parquet"
        return archive_dir / "ecmwf-ens" / f"{run_day:%Y}" / stem

    day_of = features.index.normalize().date
    run_of = pd.to_datetime(features["run_date"]).dt.date.to_numpy()
    primary_days = {d for d, r in zip(day_of, run_of, strict=True) if r == d - timedelta(days=1)}

    changed = False
    day = max(primary_days) + timedelta(days=1)
    while day <= through_run_date + timedelta(days=1):
        primary = archive_path(day - timedelta(days=1), 0)
        fallback = archive_path(day - timedelta(days=2), 12)
        if primary.exists():
            rows = run_features(primary)
            features = pd.concat([features[features.index.normalize().date != day], rows])
            changed = True
            print(f"features appended for delivery {day} (00Z run)")
        elif (day_of == day).any():
            print(f"keeping 12Z fallback features for delivery {day} (00Z still missing)")
        elif allow_fallback and fallback.exists():
            features = pd.concat([features, run_features(fallback)])
            changed = True
            print(f"features appended for delivery {day} (12Z FALLBACK run)")
        else:
            print(f"no ECMWF archive for delivery {day}")
        day += timedelta(days=1)

    if changed:
        features = features.sort_index()
        features.to_parquet(features_path)
    return features


def own_res_forecast(
    features: pd.DataFrame, dataset: pd.DataFrame, cache_dir: Path, delivery: pd.Timestamp
) -> pd.Series:
    """Train on all history before `delivery`, predict its 24 hours (aggregate MW)."""
    delivery_hours = pd.date_range(delivery, periods=24, freq="1h", tz="UTC")
    if not delivery_hours.isin(features.index).all():
        raise RuntimeError(f"ENS features for delivery day {delivery:%Y-%m-%d} missing")

    targets = dataset[list(RES_TARGETS)].dropna()
    train_index = features.index.intersection(targets.index)
    train_index = train_index[train_index < delivery]
    capacity = hourly_capacity(cache_dir, train_index.union(delivery_hours))

    def design(idx: pd.DatetimeIndex) -> pd.DataFrame:
        x = features.loc[idx].drop(columns=["run_date"])
        x["hour"] = idx.hour
        x["doy"] = idx.dayofyear
        return x

    total = pd.Series(0.0, index=delivery_hours)
    for target, cap_col in RES_TARGETS.items():
        cf = targets[target].loc[train_index] / capacity[cap_col].loc[train_index].values
        model = HistGradientBoostingRegressor(random_state=0)
        model.fit(design(train_index), cf)
        cf_pred = np.clip(model.predict(design(delivery_hours)), 0.0, None)
        total += cf_pred * capacity[cap_col].loc[delivery_hours].values
    return total


def load_surrogate_forecast(
    features: pd.DataFrame, cache_dir: Path, delivery: pd.Timestamp
) -> pd.Series:
    """Surrogate for the TSO day-ahead load forecast (registered `load-de`).

    Imitates the missing *input* series, not physical load — LEAR's weights
    were calibrated against the TSO forecast including its biases. Trained on
    all cached history before `delivery`; inputs all survive an ENTSO-E
    outage: ENS weather stats, calendar + holiday shares, and the D−7 load
    forecast (which HGB tolerates going NaN in a long outage). Measured cost
    at the price level: +0.31 EUR/MWh MAE (plan addendum 2026-08-31).
    """
    from pred_el_prices.features.holidays import holiday_share

    load_fc = resample_hourly(cache.load(cache_dir, "entsoe/load_forecast"))["Forecasted Load"]
    delivery_hours = pd.date_range(delivery, periods=24, freq="1h", tz="UTC")
    lag_w = load_fc.copy()
    lag_w.index = lag_w.index + pd.Timedelta(days=7)
    weather = [c for c in features.columns if c.startswith(("t2m_", "ssrd_"))]

    def design(idx: pd.DatetimeIndex) -> pd.DataFrame:
        x = features.loc[idx, weather].copy()
        x["load_d7"] = lag_w.reindex(idx)
        x["hour"] = idx.hour
        x["weekday"] = idx.dayofweek
        x["doy"] = idx.dayofyear
        for off in (-1, 0, 1):
            x[f"holiday_{off:+d}"] = holiday_share(idx, off)
        return x

    train_index = features.index.intersection(load_fc.dropna().index)
    train_index = train_index[train_index < delivery]
    model = HistGradientBoostingRegressor(random_state=0)
    model.fit(design(train_index), load_fc.reindex(train_index))
    pred = model.predict(design(delivery_hours)).clip(min=0.0)
    return pd.Series(pred, index=delivery_hours)


def lear_forecast(
    dataset: pd.DataFrame,
    delivery: pd.Timestamp,
    load_forecast: pd.Series,
    own_res: pd.Series,
) -> pd.Series:
    """One production LEAR day: calibrate on trailing complete days, predict `delivery`."""
    cols = {
        "price_eur_mwh": dataset["price_eur_mwh"],
        "load_forecast_mw": dataset["load_forecast_mw"],
        "res_forecast_mw": dataset[
            ["wind_onshore_forecast_mw", "wind_offshore_forecast_mw", "solar_forecast_mw"]
        ].sum(axis=1),
    }
    frame_all = pd.DataFrame(cols)

    # The last 1-2 UTC hours of the day before delivery belong to the next
    # LOCAL day: their auction runs today at 12:00 and their TSO forecasts
    # publish tonight — neither exists pre-gate. Heal that boundary with
    # 24h-lag values so the calibration window has a complete last day.
    last_hours = pd.date_range(delivery - pd.Timedelta(hours=24), periods=24, freq="1h", tz="UTC")
    frame_all = frame_all.reindex(frame_all.index.union(last_hours))
    boundary = frame_all.loc[last_hours]
    n_missing = int(boundary.isna().any(axis=1).sum())
    if 0 < n_missing <= 4:
        lagged = frame_all.reindex(last_hours - pd.Timedelta(days=1)).set_axis(last_hours)
        frame_all.loc[last_hours] = boundary.fillna(lagged)
        print(f"healed {n_missing} boundary hour(s) of the calibration frame from 24h-lag")

    hist = frame_all.dropna()
    day_sizes = hist.groupby(hist.index.normalize()).size()
    full_days = day_sizes[day_sizes == 24].index.sort_values()
    full_days = full_days[full_days < delivery]
    expected_last = delivery - pd.Timedelta(days=1)
    if full_days[-1] != expected_last:
        raise RuntimeError(
            f"dataset ends {full_days[-1]:%Y-%m-%d}, need complete {expected_last:%Y-%m-%d} "
            "(prices publish ~12:45 CET on the prior day; refresh the cache)"
        )
    window_days = full_days[-(WINDOW + LAG_BURN_IN_DAYS) :]
    hours = hist.index[pd.DatetimeIndex(hist.index.normalize()).isin(window_days)]
    frame = hist.loc[hours]

    delivery_hours = pd.date_range(delivery, periods=24, freq="1h", tz="UTC")
    target_row = pd.DataFrame(
        {
            "price_eur_mwh": np.nan,
            "load_forecast_mw": load_forecast.reindex(delivery_hours).values,
            "res_forecast_mw": own_res.reindex(delivery_hours).values,
        },
        index=delivery_hours,
    )
    if target_row[["load_forecast_mw", "res_forecast_mw"]].isna().any().any():
        raise RuntimeError(f"incomplete exog for delivery day {delivery:%Y-%m-%d}")
    frame = pd.concat([frame, target_row])

    n_days = len(window_days) + 1
    prices = frame["price_eur_mwh"].to_numpy().reshape(n_days, 24)
    exog = np.stack(
        [
            frame["load_forecast_mw"].to_numpy().reshape(n_days, 24),
            frame["res_forecast_mw"].to_numpy().reshape(n_days, 24),
        ],
        axis=2,
    )
    dow = np.array([d.dayofweek for d in [*window_days, delivery]])
    return pd.Series(forecast_day(prices, exog, dow), index=delivery_hours, name="forecast")


def write_site_json(out_dir: Path, log_path: Path, prices: pd.Series) -> None:
    """Derive latest.json and history.json from the forecast log + known prices."""
    log = pd.read_parquet(log_path)
    # An evening edition is replaced by the next morning's run: both row sets
    # stay in the append-only log, the LAST appended row per hour stands.
    log = log[~log.index.duplicated(keep="last")].sort_index()
    latest_day = log.index.normalize().max()
    hours = log.loc[log.index.normalize() == latest_day]
    actuals = prices.reindex(hours.index)

    def flag(rows: pd.DataFrame, col: str) -> bool:
        return bool(rows.get(col, pd.Series(False, index=rows.index)).fillna(False).any())

    # Honesty flags derived per run, not hardcoded: which weather vintage fed
    # the RES forecast, and whether generation actually beat the auction gate
    # (12:00 Europe/Berlin on D-1). Rows logged before 2026-08-27 lack the
    # vintage column — those were all primary-vintage mornings.
    vintage = hours["weather_vintage"].iloc[0] if "weather_vintage" in hours.columns else "00Z"
    if not isinstance(vintage, str):
        vintage = "00Z"
    gate = pd.Timestamp(f"{latest_day - pd.Timedelta(days=1):%Y-%m-%d} 12:00", tz="Europe/Berlin")
    pre_gate = pd.Timestamp(hours["generated_utc"].iloc[0]) <= gate
    is_evening = flag(hours, "evening")
    is_surrogate = flag(hours, "load_surrogate")
    note = f"{NOTE_GATE_OK if pre_gate else NOTE_GATE_MISSED} "
    note += NOTE_EVENING if is_evening else NOTE_RES[vintage]
    if is_surrogate and not is_evening:
        note += f" {NOTE_LOAD_SURROGATE}"
    latest = {
        "generated_utc": hours["generated_utc"].iloc[0],
        "delivery_day": f"{latest_day:%Y-%m-%d}",
        "model": MODEL_LABEL,
        "pre_gate": bool(pre_gate),
        "weather_vintage": vintage,
        **({"evening": True} if is_evening else {}),
        **({"load_surrogate": True} if is_surrogate else {}),
        "note": note,
        "hours": [
            {
                "t": t.isoformat(),
                "forecast": round(float(f), 2),
                "actual": None if pd.isna(a) else round(float(a), 2),
            }
            for t, f, a in zip(hours.index, hours["forecast"], actuals, strict=True)
        ],
    }

    err = (log["forecast"] - prices.reindex(log.index)).abs()
    daily = err.groupby(err.index.normalize()).agg(["mean", "count"])
    scored = daily[(daily["count"] == 24) & daily["mean"].notna()]
    days = {f"{day:%Y-%m-%d}": round(float(row["mean"]), 2) for day, row in scored.iterrows()}
    # UTC delivery blocks end 22-23h local, so a day's last 1-2 hours clear
    # only in the NEXT day's auction — "today" is stuck at 22/24 scored
    # hours all morning. Publish it anyway, flagged provisional with its
    # scored-hour count; the post-auction refresh replaces it with the full
    # score. The latest delivery day stays out of history: it lives in
    # latest.json until the next forecast supersedes it.
    partial: dict[str, int] = {}
    logged = log.groupby(log.index.normalize()).size()
    for day, row in daily[(daily["count"] > 0) & (daily["count"] < 24)].iterrows():
        if day < latest_day and logged.get(day, 0) == 24:
            days[f"{day:%Y-%m-%d}"] = round(float(row["mean"]), 2)
            partial[f"{day:%Y-%m-%d}"] = int(row["count"])
    # Scored days keep their full hourly curve so the site can show any past
    # day's forecast against the real prices, not just the MAE number.
    log_actuals = prices.reindex(log.index)
    curves = {}
    for day in days:
        mask = log.index.normalize() == pd.Timestamp(day, tz="UTC")
        hrs = log.loc[mask]
        if len(hrs) != 24:
            continue
        curves[day] = [
            {
                "t": t.isoformat(),
                "forecast": round(float(f), 2),
                "actual": None if pd.isna(a) else round(float(a), 2),
            }
            for t, f, a in zip(hrs.index, hrs["forecast"], log_actuals[mask], strict=True)
        ]
    # Merge with the published history: the log is per-run state, but scored
    # days must survive a log reseed (as on 2026-08-15, which wiped the site
    # scorecard). Freshly scored days win over previously published ones;
    # days scored before curves were published stay MAE-only. Post-gate
    # reconstructions (backfill_history) exist only in the published file —
    # their flag must ride along or the site would show them as pre-gate.
    # Fallback provenance per day, from the standing (deduped) log rows: a
    # day whose standing forecast was the evening edition or surrogate-built
    # is flagged so the site can exclude it from the headline mean (middle
    # band of the load-de wiring rule, plan addendum 2026-08-31). A day
    # generated past its own gate self-flags post_gate — recovery runs after
    # an outage must not pass as pre-gate days in history.
    post_gate: set[str] = set()
    day_flags: dict[str, dict] = {}
    for day, rows in log.groupby(log.index.normalize()):
        flags = {
            **({"evening": True} if flag(rows, "evening") else {}),
            **({"load_surrogate": True} if flag(rows, "load_surrogate") else {}),
        }
        if flags:
            day_flags[f"{day:%Y-%m-%d}"] = flags
        day_gate = pd.Timestamp(f"{day - pd.Timedelta(days=1):%Y-%m-%d} 12:00", tz="Europe/Berlin")
        if pd.Timestamp(rows["generated_utc"].iloc[0]) > day_gate:
            post_gate.add(f"{day:%Y-%m-%d}")
    history_path = out_dir / "history.json"
    if history_path.exists():
        for entry in json.loads(history_path.read_text(encoding="utf-8"))["days"]:
            if entry["day"] not in days:
                days[entry["day"]] = entry["mae"]
                if "partial" in entry:
                    partial[entry["day"]] = entry["partial"]
                if "hours" in entry:
                    curves[entry["day"]] = entry["hours"]
                if entry.get("post_gate"):
                    post_gate.add(entry["day"])
                merged = {
                    **({"evening": True} if entry.get("evening") else {}),
                    **({"load_surrogate": True} if entry.get("load_surrogate") else {}),
                }
                if merged:
                    day_flags[entry["day"]] = merged
    history = {
        "days": [
            {
                "day": d,
                "mae": m,
                **({"partial": partial[d]} if d in partial else {}),
                **({"post_gate": True} if d in post_gate else {}),
                **day_flags.get(d, {}),
                **({"hours": curves[d]} if d in curves else {}),
            }
            for d, m in sorted(days.items())[-60:]
        ]
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest.json").write_text(json.dumps(latest, indent=1), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")


def backfill_history(out_dir: Path, run_dirs: list[Path], start: str, end: str) -> int:
    """Fill curve-less history days from backtest forecasts, flagged post-gate.

    One-off recovery for days the live workflow never forecast, or whose
    curves were lost to a log reseed (the seeded MAEs themselves came from a
    backtest run, so scoring the same reconstruction is consistent with the
    published numbers). Each missing day takes its 24 forecast/actual hours
    from the first run in `run_dirs` that covers it fully, gets its MAE
    recomputed from that curve, and is published flagged `post_gate` — these
    are computed after the auction, not pre-gate forecasts. Days already
    carrying a curve are left untouched.
    """
    history_path = out_dir / "history.json"
    entries = {e["day"]: e for e in json.loads(history_path.read_text(encoding="utf-8"))["days"]}
    frames = [pd.read_parquet(d / "forecast.parquet") for d in run_dirs]
    filled = 0
    for day in pd.date_range(start, end, tz="UTC"):
        key = f"{day:%Y-%m-%d}"
        if "hours" in entries.get(key, {}):
            continue
        for frame in frames:
            rows = frame[frame.index.normalize() == day]
            if len(rows) == 24 and rows["actual"].notna().all():
                break
        else:
            print(f"{key}: no run covers the day; left as is")
            continue
        mae = round(float((rows["lear_forecast"] - rows["actual"]).abs().mean()), 2)
        old = entries.get(key)
        if old is not None and old["mae"] != mae:
            print(f"{key}: published MAE {old['mae']} replaced by recomputed {mae}")
        entries[key] = {
            "day": key,
            "mae": mae,
            "post_gate": True,
            "hours": [
                {"t": t.isoformat(), "forecast": round(float(f), 2), "actual": round(float(a), 2)}
                for t, f, a in zip(rows.index, rows["lear_forecast"], rows["actual"], strict=True)
            ],
        }
        filled += 1
    history = {"days": [entries[d] for d in sorted(entries)][-60:]}
    history_path.write_text(json.dumps(history, indent=1), encoding="utf-8")
    print(f"backfilled {filled} day(s); history holds {len(history['days'])}")
    return filled


def run_daily(
    cache_dir: Path,
    archive_dir: Path,
    features_path: Path,
    out_dir: Path,
    delivery_day=None,
    skip_fetch: bool = False,
    allow_ens_fallback: bool = False,
    refresh_only: bool = False,
    evening: bool = False,
    allow_load_surrogate: bool = False,
) -> Path | None:
    """Produce and publish the forecast for the next UTC day. Idempotent per day.

    `evening` is the evening edition (plan addendum 2026-08-31): runs the
    night before the normal slot, targets the day AFTER tomorrow, builds on
    the just-archived 12Z ENS run plus the load surrogate (the TSO load
    forecast for that day does not exist yet), and is replaced by the next
    morning's regular run. `allow_load_surrogate` lets a morning retry slot
    publish with the surrogate when ENTSO-E has no load forecast (measured
    cost ~+0.3 EUR/MWh); the first slot stays strict so real data gets its
    chance to arrive.
    """
    now = datetime.now(UTC)
    if evening:
        allow_ens_fallback = True  # the 12Z vintage IS the evening's weather
        allow_load_surrogate = True
    delivery = (
        pd.Timestamp(delivery_day, tz="UTC")
        if delivery_day
        else pd.Timestamp(now.date() + timedelta(days=2 if evening else 1), tz="UTC")
    )
    run_date = (delivery - pd.Timedelta(days=1)).date()
    log_path = out_dir / "forecast_log.parquet"

    if refresh_only:
        # Post-auction slot: auction results publish ~12:45 CET/CEST, so a
        # price refresh now fills the actuals next to the morning forecast
        # and scores newly completed days. Never forecasts — a "forecast"
        # generated after the results are public would be worthless even
        # flagged, so a missed morning run stays an honest gap.
        if not log_path.exists():
            print("refresh-only: no forecast log; nothing to refresh")
            return None
        if not skip_fetch:
            import requests
            from entsoe import EntsoePandasClient

            from pred_el_prices.config import entsoe_api_key
            from pred_el_prices.pipeline.entsoe import backfill

            client = EntsoePandasClient(api_key=entsoe_api_key())
            # Best-effort like the forecast path: during a platform outage
            # there are no new prices to fetch, so rewriting the site JSON
            # from cache is the correct (idempotent) outcome, not a failure.
            try:
                backfill(
                    client,
                    ["day_ahead_prices"],
                    pd.Timestamp("2015-01-01", tz="UTC"),
                    delivery + pd.Timedelta(days=1),
                    cache_dir,
                )
            except requests.RequestException as e:
                print(f"WARN: ENTSO-E refresh failed ({e}); rewriting from cached prices")
        prices = resample_hourly(cache.load(cache_dir, "entsoe/day_ahead_prices"))["price_eur_mwh"]
        write_site_json(out_dir, log_path, prices)
        print("refresh-only: site JSON rewritten with current prices")
        return out_dir / "latest.json"

    if not skip_fetch:
        import requests
        from entsoe import EntsoePandasClient

        from pred_el_prices.config import entsoe_api_key
        from pred_el_prices.pipeline.capacity import update_cache
        from pred_el_prices.pipeline.ecmwf import archive_run
        from pred_el_prices.pipeline.entsoe import backfill

        client = EntsoePandasClient(api_key=entsoe_api_key())
        # Best-effort: during a platform outage (2026-08-30/31: full 503 for
        # days) the fetch must not kill the run — the caches carry enough
        # history to forecast, and whatever is genuinely missing fails its
        # own specific check further down instead of dying here.
        try:
            backfill(
                client,
                ["day_ahead_prices", "load_forecast", "wind_solar_forecast"],
                pd.Timestamp("2015-01-01", tz="UTC"),
                delivery + pd.Timedelta(days=1),
                cache_dir,
            )
        except requests.RequestException as e:
            print(f"WARN: ENTSO-E refresh failed ({e}); proceeding on cached data")
        update_cache(cache_dir)
        # Best-effort: a failure must not kill the run — the 12Z fallback
        # may cover the day. Gap healing of older 00Z runs happens in the
        # evening archive-ens-12z workflow, outside the morning S3 herd.
        try:
            archive_run(run_date, archive_dir)
        except (FileNotFoundError, requests.RequestException) as e:
            print(f"WARN: 00Z ENS run {run_date} unavailable: {e}")

    features = update_features(features_path, archive_dir, run_date, allow_ens_fallback)
    dataset, _ = build_dataset(cache_dir)
    prices = resample_hourly(cache.load(cache_dir, "entsoe/day_ahead_prices"))["price_eur_mwh"]

    if log_path.exists():
        logged = pd.read_parquet(log_path)
        day_rows = logged[logged.index.normalize() == delivery]
        if len(day_rows):
            # Evening rows are a preview: a morning run replaces them (the
            # log stays append-only — scoring keeps the last row per hour).
            ev = day_rows.get("evening", pd.Series(False, index=day_rows.index)).fillna(False)
            if evening or (~ev).any():
                print(f"forecast for {delivery:%Y-%m-%d} already logged; refreshing site JSON only")
                write_site_json(out_dir, log_path, prices)
                return None
            print(f"evening edition for {delivery:%Y-%m-%d} logged; this run replaces it")

    load_fc = resample_hourly(cache.load(cache_dir, "entsoe/load_forecast"))["Forecasted Load"]
    delivery_hours = pd.date_range(delivery, periods=24, freq="1h", tz="UTC")
    load_d = load_fc.reindex(delivery_hours)
    missing = load_d.isna()
    load_surrogate = False
    if missing.sum() > 4:
        if not allow_load_surrogate:
            raise RuntimeError(
                f"ENTSO-E load forecast for {delivery:%Y-%m-%d} not yet published; retry later"
            )
        load_d = load_surrogate_forecast(features, cache_dir, delivery)
        load_surrogate = True
        print(f"load input: surrogate model (TSO forecast for {delivery:%Y-%m-%d} unavailable)")
    elif missing.any():
        # ENTSO-E publishes local (CET/CEST) days: the last 1-2 hours of the
        # UTC delivery block belong to the next local day and do not exist
        # pre-gate. Fill from 24 h earlier (published, flat night load).
        lagged = load_fc.reindex(delivery_hours - pd.Timedelta(days=1))
        load_d = load_d.fillna(pd.Series(lagged.to_numpy(), index=delivery_hours))
        print(f"filled {int(missing.sum())} boundary hour(s) of load forecast from 24h-lag")

    # Multi-day outage: ENTSO-E gaps in the trailing days would starve LEAR's
    # calibration lags (exog at d, d-1, d-7) even though prices kept flowing
    # through the refresh slots. Fill up to 7 such days with the same
    # pre-gate substitutes the delivery day uses — the surrogate load and the
    # own-RES forecast (written into the onshore column; LEAR's academic
    # config only ever reads the three RES columns summed).
    if allow_load_surrogate:
        res_cols = list(RES_TARGETS)
        for day in pd.date_range(delivery - pd.Timedelta(days=7), delivery - pd.Timedelta(days=1)):
            hrs = pd.date_range(day, periods=24, freq="1h", tz="UTC")
            sub = dataset.reindex(hrs)
            # a day's last 1-2 price hours clear only the NEXT day (local-day
            # boundary) — lear_forecast heals those; demand the rest
            if sub["price_eur_mwh"].notna().sum() < 20:
                continue
            # the boundary hours may be absent as rows entirely (the price
            # series defines the index) — they must exist to take fill values
            dataset = dataset.reindex(dataset.index.union(hrs))
            sub = dataset.loc[hrs]
            if not hrs.isin(features.index).all():
                continue
            if sub["load_forecast_mw"].isna().any():
                dataset.loc[hrs, "load_forecast_mw"] = load_surrogate_forecast(
                    features, cache_dir, day
                ).to_numpy()
                print(f"calibration gap {day:%Y-%m-%d}: load forecast filled by surrogate")
            if sub[res_cols].isna().any().any():
                total = own_res_forecast(features, dataset, cache_dir, day)
                dataset.loc[hrs, res_cols[0]] = total.to_numpy()
                dataset.loc[hrs, res_cols[1:]] = 0.0
                print(f"calibration gap {day:%Y-%m-%d}: RES forecast filled by own-RES")

    own_res = own_res_forecast(features, dataset, cache_dir, delivery)
    forecast = lear_forecast(dataset, delivery, load_d, own_res)

    day_runs = pd.to_datetime(features.loc[delivery_hours, "run_date"]).dt.date
    entry = forecast.to_frame()
    entry["generated_utc"] = now.isoformat(timespec="seconds")
    entry["weather_vintage"] = "00Z" if (day_runs == run_date).all() else "12Z"
    entry["evening"] = evening
    entry["load_surrogate"] = load_surrogate
    if log_path.exists():
        entry = pd.concat([pd.read_parquet(log_path), entry])
    out_dir.mkdir(parents=True, exist_ok=True)
    entry.to_parquet(log_path)

    write_site_json(out_dir, log_path, prices)
    print(f"forecast for {delivery:%Y-%m-%d} written to {out_dir}")
    return out_dir / "latest.json"
