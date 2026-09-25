"""Morning vs evening configuration of the production networks (`pep evening-nets`, P53-P55).

Per delivery day D in [start, end], from ONE weekly bundle (trained once per week, as
the backfill does), two forecasts:
- morning (09:15 UTC on D-1): own RES from the 00Z ENS run of D-1, TSO load forecasts
  (DE and neighbours), 15-minute shape on the TSO 15-min load: exactly backfill-nets;
- evening (21:35 UTC on D-2): own RES from the 12Z run of D-2, DE load from the load-de
  surrogate, neighbour load from nets.neighbour_load_surrogate, 15-minute shape on the
  interpolated DE surrogate.
Both use the same PIT history: the seed before `start` plus the morning log as it grows
(the morning forecast replaces the evening one in production, so it is what the PIT
window holds). The dataset is cut at D-1 22:00 UTC for both: every lag the evening row
uses (D-1 prices UTC 00-21, TSO RES and loads of D-1) was published by 21:35 on D-2.
Known small optimism: the DE load surrogate trains on the cached TSO load up to D, whose
UTC 22-23 of D-1 publish only on D-1 (2 of ~20,000 training hours).

Output: morning/ and evening/ nets logs (single files), bundles/, summary.json.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pred_el_prices.daily_forecast import (
    load_surrogate_forecast,
    own_res_parts,
    site_prices,
)
from pred_el_prices.features.dataset import NEIGHBOUR_ZONES, build_dataset
from pred_el_prices.models.qnn import load_bundle, save_bundle
from pred_el_prices.pipeline import cache
from pred_el_prices.pipeline.entsoe import resample_hourly
from pred_el_prices.production import nets
from pred_el_prices.production import site as nets_site
from pred_el_prices.production.backfill import scores


def _daily_pinball(log: pd.DataFrame, kind: str, actual: pd.Series) -> pd.Series:
    rows = log[log["kind"] == kind]
    a = actual.reindex(rows.index)
    rows, a = rows[a.notna()], a[a.notna()]
    pb = nets_site._pinball_rows(rows[nets_site.Q_COLS].to_numpy(float), a.to_numpy())
    return pd.Series(pb, index=rows.index).groupby(rows.index.normalize()).mean()


def dm(a: pd.Series, b: pd.Series) -> float:
    """Diebold-Mariano on daily losses, a - b (positive: a worse)."""
    d = (a - b).dropna()
    return round(float(d.mean() / np.sqrt(d.var(ddof=0) / len(d))), 2)


def run(
    start: str,
    end: str,
    cache_dir: Path,
    features_path: Path,
    features12_path: Path,
    seed_path: Path,
    out_dir: Path,
    lear_log: Path | None = None,
    n_jobs: int = -1,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bundles").mkdir(exist_ok=True)
    first = pd.Timestamp(start, tz="UTC")
    seed = pd.read_parquet(seed_path)
    seed_file = out_dir / nets_site.SEED_NAME
    seed[seed.index < first].to_parquet(seed_file)
    logs = {c: out_dir / c / nets_site.LOG_NAME for c in ("morning", "evening")}
    for p in logs.values():
        p.parent.mkdir(exist_ok=True)

    dataset, _ = build_dataset(cache_dir)
    f00 = pd.read_parquet(features_path)
    f12 = pd.read_parquet(features12_path)
    prices = site_prices(cache_dir)
    qh = nets.QHInputs.from_cache(cache_dir)
    done = {
        c: set(nets_site.read_log(p).index.normalize()) if p.exists() else set()
        for c, p in logs.items()
    }

    subs, timing, skipped, loaded = [], {"train_s": {}, "day_s": []}, [], {}
    for day in pd.date_range(first, pd.Timestamp(end, tz="UTC"), freq="D"):
        hours = pd.date_range(day, periods=24, freq="1h")
        if not (hours.isin(f00.index).all() and hours.isin(f12.index).all()):
            skipped.append(f"{day:%Y-%m-%d}")
            print(f"{day:%Y-%m-%d}: 00Z or 12Z ENS features missing; skipped", flush=True)
            continue
        if day in done["morning"] and day in done["evening"]:
            continue
        monday = nets.monday_of(day)
        bundle = out_dir / "bundles" / f"nets-{monday:%Y-%m-%d}.pt"
        if not bundle.exists():
            t0 = time.perf_counter()
            fitted, meta = nets.train_ensemble(dataset, monday, n_jobs=n_jobs)
            save_bundle(bundle, fitted, meta)
            timing["train_s"][f"{monday:%Y-%m-%d}"] = round(time.perf_counter() - t0, 1)
        if bundle not in loaded:
            loaded = {bundle: load_bundle(bundle)}
        networks, meta = loaded[bundle]

        t0 = time.perf_counter()
        seen = dataset[dataset.index < day - pd.Timedelta(hours=2)]
        history = nets.load_history(seed_file, logs["morning"])
        # morning: exactly the backfill's day
        parts = own_res_parts(f00, seen, cache_dir, day)
        h_m, q_m, fl_m = nets.forecast_day(
            networks, meta, day, seen, cache_dir, parts, history, prices, qh
        )
        # evening: the 00Z history before D plus the 12Z run of D-2 for D itself
        f_e = pd.concat([f00[f00.index < day], f12.loc[hours]]).sort_index()
        parts_e = own_res_parts(f_e, seen, cache_dir, day)
        load_de = load_surrogate_forecast(f_e, cache_dir, day)
        load_nb = nets.neighbour_load_surrogate(f_e, cache_dir, day)
        h_e, q_e, fl_e = nets.forecast_day(
            networks, meta, day, seen, cache_dir, parts_e, history, prices, qh,
            load_de, load_nb, evening=True,
        )  # fmt: skip
        now = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
        for c, h, q, fl in (("morning", h_m, q_m, fl_m), ("evening", h_e, q_e, fl_e)):
            nets_site.append_log(logs[c], nets_site.log_rows(h, q, {**fl, "backfill": True}, now))

        # the substitutes against what the TSOs published later
        tso_de = resample_hourly(cache.load(cache_dir, "entsoe/load_forecast"))[
            "Forecasted Load"
        ].reindex(hours)
        tso_nb = sum(
            resample_hourly(cache.load(cache_dir, f"entsoe/{z}/load_forecast"))[
                "Forecasted Load"
            ].reindex(hours)
            for z in NEIGHBOUR_ZONES
        )
        subs.append(
            pd.DataFrame(
                {
                    "de_sub": load_de.to_numpy(),
                    "de_tso": tso_de.to_numpy(),
                    "nb_sub": load_nb.to_numpy(),
                    "nb_tso": tso_nb.to_numpy(),
                    "res_12z": parts_e.sum(axis=1).to_numpy(),
                    "res_00z": parts.sum(axis=1).to_numpy(),
                },
                index=hours,
            )
        )
        timing["day_s"].append(round(time.perf_counter() - t0, 1))
        print(f"{day:%Y-%m-%d}: done ({timing['day_s'][-1]} s)", flush=True)

    if subs:
        sub_path = out_dir / "substitutes.parquet"
        if sub_path.exists():  # a resumed run keeps the days done before
            subs.insert(0, pd.read_parquet(sub_path))
        s = pd.concat(subs)
        s[~s.index.duplicated(keep="last")].sort_index().to_parquet(sub_path)
    summary = summarize(out_dir, cache_dir, prices, lear_log)
    summary.update({"window": [start, end], "skipped_days": skipped, "timing": timing})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def summarize(out_dir: Path, cache_dir: Path, prices: pd.Series, lear_log=None) -> dict:
    """Scores of both configurations on the days both have, DM, substitute errors."""
    prices_qh = nets_site.quarter_prices(cache_dir)
    m = nets_site.read_log(out_dir / "morning" / nets_site.LOG_NAME)
    e = nets_site.read_log(out_dir / "evening" / nets_site.LOG_NAME)
    common = m.index.normalize().unique().intersection(e.index.normalize().unique())
    m, e = m[m.index.normalize().isin(common)], e[e.index.normalize().isin(common)]
    lear = None
    if lear_log is not None:
        lear = pd.read_parquet(lear_log)["forecast"]
    out = {
        "morning": scores(m, prices, prices_qh, lear),
        "evening": scores(e, prices, prices_qh, lear),
    }
    out["dm_evening_minus_morning"] = {
        "hourly": dm(_daily_pinball(e, "h", prices), _daily_pinball(m, "h", prices)),
        "quarter": dm(_daily_pinball(e, "qh", prices_qh), _daily_pinball(m, "qh", prices_qh)),
    }
    mh, eh = out["morning"]["hourly"], out["evening"]["hourly"]
    out["evening_cost"] = {
        "pinball_pct": round(100 * (eh["pinball"] / mh["pinball"] - 1), 2),
        "mae_eur": round(eh["mae"] - mh["mae"], 3),
    }
    if "quarter" in out["morning"]:
        mq, eq = out["morning"]["quarter"], out["evening"]["quarter"]
        out["evening_cost"]["pinball_qh_pct"] = round(100 * (eq["pinball"] / mq["pinball"] - 1), 2)
    sub_path = out_dir / "substitutes.parquet"
    if sub_path.exists():
        s = pd.read_parquet(sub_path).dropna()
        out["substitutes"] = {
            "hours": len(s),
            "nb_mae_mw": round(float((s["nb_sub"] - s["nb_tso"]).abs().mean()), 1),
            "nb_mae_pct": round(
                float(100 * (s["nb_sub"] - s["nb_tso"]).abs().mean() / s["nb_tso"].mean()), 2
            ),
            "de_mae_mw": round(float((s["de_sub"] - s["de_tso"]).abs().mean()), 1),
            "de_mae_pct": round(
                float(100 * (s["de_sub"] - s["de_tso"]).abs().mean() / s["de_tso"].mean()), 2
            ),
            "own_res_12z_vs_00z_mad_mw": round(
                float((s["res_12z"] - s["res_00z"]).abs().mean()), 1
            ),
        }
    return out
