"""Production replay of the quantile networks (`qnn-replay`): gate-available inputs only.

Trains as the backtest does (weekly refits, rolling window, fuel scaling, neighbour
LOAD instead of residual load, since neighbour wind/solar forecasts publish after the
gate), but predicts each delivery day D from what production has at the gate:
- German wind+solar for D: production's own RES forecast (ECMWF ENS 00Z, replayed
  daily, `own_res_path`), not the TSO forecast (published 18:00 D-1).
- Load (DE and neighbours) for D: the TSO day-ahead forecast, except UTC hours
  22-23, which belong to the next local day and are filled from 24 h earlier.
Lags (d-1, d-7) keep the TSO values: those were published before D's gate.
Each day is also predicted from the TSO inputs, to show what gate-availability costs.

Artifacts: replay.npz with q_prod and q_tso (head, seed, n_hours, 99) EUR/MWh
(unclipped, unsorted across seeds), heads, index (UTC ns), actual.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pred_el_prices.experiments.qnn_de import design, fuel_cost, load_days
from pred_el_prices.models.qnn import QNNConfig, fit_predict


def _fit(x, y, row_days, x_pred, start, config, seed, window_days):
    import torch

    torch.set_num_threads(1)
    last_train = start - pd.Timedelta(days=2)
    train = (row_days <= last_train) & (row_days > last_train - pd.Timedelta(days=window_days))
    return fit_predict(x[train], y[train], x_pred, 7, config, seed)


def run(
    out_dir: Path,
    start: str = "2026-07-20",
    end: str = "2026-09-22",
    own_res_path: str = "runs/own-res/own_res.parquet",
    dataset_path: str = "data/dataset/hourly.parquet",
    train_start: str = "2018-12-01",
    n_seeds: int = 12,
    heads: list[str] | None = None,
    hidden: list[int] | None = None,
    dropout: float = 0.2784723989915054,
    lr: float = 0.0004983021010966105,
    weight_decay: float = 8.548006178626775e-05,
    batch_size: int = 32,
    window_days: int = 1460,
    n_jobs: int = -1,
) -> dict:
    heads = heads or ["quantile", "jsu"]
    prices, exog, fuels, days, hours = load_days(dataset_path, train_start, [], neighbours="load")
    scale = fuel_cost(fuels)
    prices_s = prices / scale[:, None]
    x_tso, y, row_days = design(prices_s, exog, fuels, days)

    own = pd.read_parquet(own_res_path)["own_res_mw"]
    window = pd.date_range(start, end, freq="D", tz="UTC")
    window = window[window.isin(days)]
    # a day without the ENS archive had no production forecast at all: skip it
    has_own = [own.reindex(pd.date_range(d, periods=24, freq="1h")).notna().all() for d in window]
    skipped = [f"{d:%Y-%m-%d}" for d, ok in zip(window, has_own, strict=True) if not ok]
    window = window[np.array(has_own)]
    if skipped:
        print(f"no own RES forecast (ENS archive gap), skipped: {skipped}", flush=True)
    x_prod = np.empty((len(window), x_tso.shape[1]))
    for i, d in enumerate(window):
        k = days.get_loc(d)
        e = exog.copy()
        e[k, :, 1] = own.reindex(pd.date_range(d, periods=24, freq="1h")).to_numpy()
        e[k, 22:24, 0] = exog[k - 1, 22:24, 0]  # DE load: next local day, not yet published
        e[k, 22:24, 2] = exog[k - 1, 22:24, 2]  # neighbour load: same
        x_all, _, _ = design(prices_s, e, fuels, days)
        x_prod[i] = x_all[k - 7]
    in_window = row_days.isin(window)
    x_tso_w = x_tso[in_window]

    starts = pd.date_range(window[0], window[-1], freq="7D")
    base = {
        "hidden": hidden or [256, 256, 256],
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
    }
    jobs = []
    for s in starts:
        sel = (window >= s) & (window < s + pd.Timedelta(days=7))
        x_pred = np.vstack([x_prod[sel], x_tso_w[sel]])
        for h in heads:
            for k in range(n_seeds):
                jobs.append((s, sel, h, k, x_pred))
    print(
        f"{len(starts)} refits x {len(heads)} heads x {n_seeds} seeds = {len(jobs)} fits",
        flush=True,
    )
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_fit)(x_tso, y, row_days, xp, s, QNNConfig(**base, head=h), k, window_days)
        for s, _, h, k, xp in jobs
    )

    n = len(window)
    q_prod = np.full((len(heads), n_seeds, n, 24, 99), np.nan)
    q_tso = np.full_like(q_prod, np.nan)
    w_scale = scale[days.isin(window)]
    for (_, sel, h, k, _), pred in zip(jobs, results, strict=True):
        m = int(sel.sum())
        f = w_scale[sel][:, None, None]
        q_prod[heads.index(h), k, sel] = pred[:m] * f
        q_tso[heads.index(h), k, sel] = pred[m:] * f
    idx = pd.DatetimeIndex([d + pd.Timedelta(hours=hh) for d in window for hh in range(24)])
    actual = pd.Series(prices.reshape(-1), index=hours).reindex(idx).to_numpy()
    np.savez_compressed(
        out_dir / "replay.npz",
        q_prod=q_prod.reshape(len(heads), n_seeds, -1, 99).astype(np.float32),
        q_tso=q_tso.reshape(len(heads), n_seeds, -1, 99).astype(np.float32),
        heads=np.array(heads),
        index=idx.as_unit("ns").asi8,
        actual=actual,
    )
    return {
        "window": [str(window[0]), str(window[-1])],
        "days": n,
        "skipped_days": skipped,
        "fits": len(jobs),
    }
