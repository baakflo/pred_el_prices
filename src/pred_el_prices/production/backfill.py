"""Historical replay of the production network path (`pep backfill-nets`).

For each delivery day D in [start, end]: the weekly bundle trained as of the Monday of
D's week (days <= Monday - 2, the backtest convention; trained once, reused from
<out>/bundles/), our own RES forecast refit on history before D from the production
ENS feature table, the dataset cut at D-1 22:00 UTC (the auction for UTC 22-23 of D-1
runs at D's gate), and the same forecast_day() the daily job runs. The PIT history is
the seed before `start` plus the backfill's own log as it grows, as it would have been
live. Days without ENS features had no production forecast and are skipped.

Output (<out>): nets_log/ (monthly partitions; an older single nets_log.parquet there is
migrated on first read), bundles/, own_res.parquet (per-day own RES, reused on reruns),
summary.json, and, given the LEAR log/history, latest.json + history.json + days/ (v2).
Rows are flagged backfill=True with the real generation time; the site JSON publishes
them with "replay": true in every nets block.
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pred_el_prices.daily_forecast import own_res_parts, site_prices, write_site_json
from pred_el_prices.features.dataset import build_dataset
from pred_el_prices.models.qnn import load_bundle, save_bundle
from pred_el_prices.production import nets
from pred_el_prices.production import site as nets_site


def run(
    start: str,
    end: str,
    cache_dir: Path,
    features_path: Path,
    out_dir: Path,
    seed_path: Path,
    lear_log: Path | None = None,
    lear_history: Path | None = None,
    n_jobs: int = -1,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundles = out_dir / "bundles"
    bundles.mkdir(exist_ok=True)
    log_path = out_dir / nets_site.LOG_DIR
    first = pd.Timestamp(start, tz="UTC")

    # the seed may reach into the window (a backtest run does): keep only its past
    seed = pd.read_parquet(seed_path)
    seed_file = out_dir / nets_site.SEED_NAME
    seed[seed.index < first].to_parquet(seed_file)

    dataset, _ = build_dataset(cache_dir)
    features = pd.read_parquet(features_path)
    prices = site_prices(cache_dir)
    qh = nets.QHInputs.from_cache(cache_dir)
    own_path = out_dir / "own_res.parquet"
    own_all = pd.read_parquet(own_path) if own_path.exists() else pd.DataFrame()
    done = set()
    if nets_site.log_exists(log_path):
        log = nets_site.read_log(log_path)
        done = set(log.index[log["kind"] == "h"].normalize())

    timing = {"train_s": {}, "predict_s": []}
    skipped, loaded = [], {}
    for day in pd.date_range(first, pd.Timestamp(end, tz="UTC"), freq="D"):
        hours = pd.date_range(day, periods=24, freq="1h")
        if not hours.isin(features.index).all():
            skipped.append(f"{day:%Y-%m-%d}")
            print(f"{day:%Y-%m-%d}: no ENS features (no production forecast existed); skipped")
            continue
        if day in done:
            continue
        monday = nets.monday_of(day)
        bundle = bundles / f"nets-{monday:%Y-%m-%d}.pt"
        if not bundle.exists():
            t0 = time.perf_counter()
            fitted, meta = nets.train_ensemble(dataset, monday, n_jobs=n_jobs)
            save_bundle(bundle, fitted, meta)
            timing["train_s"][f"{monday:%Y-%m-%d}"] = round(time.perf_counter() - t0, 1)
            print(f"bundle {bundle.name}: {timing['train_s'][f'{monday:%Y-%m-%d}']} s")
        if bundle not in loaded:
            loaded = {bundle: load_bundle(bundle)}
        networks, meta = loaded[bundle]

        t0 = time.perf_counter()
        seen = dataset[dataset.index < day - pd.Timedelta(hours=2)]
        if len(own_all) and hours.isin(own_all.index).all():
            parts = own_all.loc[hours]
        else:
            parts = own_res_parts(features, seen, cache_dir, day)
            own_all = pd.concat([own_all, parts]).sort_index()
            own_all.to_parquet(own_path)
        history = nets.load_history(seed_file, log_path)
        hourly, quarters, flags = nets.forecast_day(
            networks, meta, day, seen, cache_dir, parts, history, prices, qh
        )
        runs = pd.to_datetime(features.loc[hours, "run_date"]).dt.date
        vintage = "00Z" if (runs == (day - pd.Timedelta(days=1)).date()).all() else "12Z"
        now = datetime.now(UTC).isoformat(timespec="seconds")
        flags = {**flags, "weather_vintage": vintage, "backfill": True}
        nets_site.append_log(log_path, nets_site.log_rows(hourly, quarters, flags, now))
        timing["predict_s"].append(round(time.perf_counter() - t0, 1))
        print(f"{day:%Y-%m-%d}: PIT days {flags['recal_days']}, shaped {flags['shaped']}")

    lear = None
    if lear_log is not None:
        lear = build_site(log_path, lear_log, cache_dir, out_dir, lear_history, end, prices)
    summary = {
        "window": [start, end],
        "skipped_days": skipped,
        "timing": {
            **timing,
            "predict_s_mean": round(float(np.mean(timing["predict_s"])), 1)
            if timing["predict_s"]
            else None,
        },
        "scores": scores(
            nets_site.read_log(log_path),
            prices,
            nets_site.quarter_prices(cache_dir),
            lear,
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def build_site(
    nets_log: Path,
    lear_log: Path,
    cache_dir: Path,
    out_dir: Path,
    history: Path | None = None,
    end: str | None = None,
    prices: pd.Series | None = None,
) -> pd.Series | None:
    """Site JSON (latest, history, days/) from an existing nets log + LEAR log + caches.

    No networks run: this is how a backfill's outputs are (re)published. The nets log
    (a single file or a partition directory) is written into `out_dir/nets_log/` unless
    it already is that directory; the LEAR log is cut after `end` so latest.json is the
    last day; `history` (a published history.json) is the base the history merges into.
    Returns the LEAR forecast series used (None if the log is empty up to `end`).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stop = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) if end is not None else None
    dest = out_dir / nets_site.LOG_DIR
    if Path(nets_log).resolve() != dest.resolve():
        log = nets_site.read_log(nets_log)
        nets_site.write_log(dest, log if stop is None else log[log.index < stop])
    rows = pd.read_parquet(lear_log)
    if stop is not None:
        rows = rows[rows.index < stop]
    if rows.empty:
        return None
    rows.to_parquet(out_dir / "forecast_log.parquet")
    if history is not None:
        # a published history newer than `end` would put days after latest.json's
        published = json.loads(Path(history).read_text(encoding="utf-8"))
        if end is not None:
            published["days"] = [e for e in published["days"] if e["day"] <= end]
        (out_dir / "history.json").write_text(json.dumps(published, indent=1), encoding="utf-8")
    if prices is None:
        prices = site_prices(cache_dir)
    write_site_json(
        out_dir, out_dir / "forecast_log.parquet", prices, nets_site.quarter_prices(cache_dir)
    )
    return rows["forecast"]


def scores(log: pd.DataFrame, prices: pd.Series, prices_qh: pd.Series, lear=None) -> dict:
    """Period scores of the logged nets (hourly, 15-min, 15-min repeated-hourly baseline)
    and, where the LEAR log has the same hours, LEAR's MAE next to the nets' on those."""
    Q = nets_site.Q_COLS
    h = log[log["kind"] == "h"]
    qh = log[log["kind"] == "qh"]
    a_h = prices.reindex(h.index)
    h = h[a_h.notna()]
    a_h = a_h[a_h.notna()]
    a_q = prices_qh.reindex(qh.index)
    qh = qh[a_q.notna()]
    a_q = a_q[a_q.notna()]

    def block(rows, a):
        q = rows[Q].to_numpy(dtype=float)
        av = a.to_numpy()
        return {
            "n": len(av),
            "mae": round(float(np.abs(q[:, 49] - av).mean()), 3),
            "pinball": round(float(nets_site._pinball_rows(q, av).mean()), 3),
            "cov80": round(float(((av >= q[:, 9]) & (av <= q[:, 89])).mean()), 4),
            "cov98": round(float(((av >= q[:, 0]) & (av <= q[:, 98])).mean()), 4),
        }

    out = {"days": int(h.index.normalize().nunique()), "hourly": block(h, a_h)}
    if len(qh):
        out["quarter"] = block(qh, a_q)
        rep = h[Q].reindex(qh.index.floor("h")).set_axis(qh.index)
        ok = rep.notna().all(axis=1)
        out["quarter_repeated_hourly"] = block(rep[ok], a_q[ok])
    if lear is not None:
        lear = lear[~lear.index.duplicated(keep="last")].reindex(h.index)
        both = lear.notna()
        full = both.groupby(both.index.normalize()).transform("all")
        idx = h.index[full.to_numpy()]
        if len(idx):
            out["vs_lear"] = {
                "days": int(idx.normalize().nunique()),
                "lear_mae": round(float((lear[idx] - a_h[idx]).abs().mean()), 3),
                "nets_mae": round(float((h.loc[idx, "q50"] - a_h[idx]).abs().mean()), 3),
            }
    return out
