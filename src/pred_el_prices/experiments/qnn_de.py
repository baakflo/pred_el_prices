"""Quantile neural network on DE-LU (`qnn-de`): 99 percentiles per hour, monthly refits.

Expanding window from `train_start`, one refit per calendar month from
`first_fit`, `n_seeds` networks per refit averaged percentile by percentile
(then sorted). Leakage: inputs are LEAR's gate-safe blocks (models.lear.build_xy)
plus fuel settlements already lagged 2 days in the dataset. A refit for month M
trains on days up to M-2 only: UTC hours 22-23 of day M-1 are local day M,
cleared in the auction being forecast, so day M-1 cannot be a training target
at the gate of M's first day.

Artifacts: quantiles.parquet (q01..q99 + actual), forecast.parquet (median +
actual, the point view that eval.scorecard reads), metrics.json.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pred_el_prices.eval.metrics import mae, naive_forecast
from pred_el_prices.models.lear import build_xy
from pred_el_prices.models.qnn import QUANTILES, QNNConfig, fit_predict

PRICE_COL = "price_eur_mwh"
RES_COLS = ["wind_onshore_forecast_mw", "wind_offshore_forecast_mw", "solar_forecast_mw"]
NEIGHBOUR_COLS = [
    "rl_fr_mw", "rl_nl_mw", "rl_be_mw", "rl_at_mw", "rl_pl_mw",
    "rl_cz_mw", "rl_ch_mw", "rl_dk_1_mw", "rl_dk_2_mw",
]  # fmt: skip
Q_COLS = [f"q{round(q * 100):02d}" for q in QUANTILES]


def load_days(dataset_path: str, train_start: str, exog_extra: list[str]):
    """Contiguous full UTC days: prices (n, 24), exog (n, 24, k), fuels (n, 2), day index."""
    ds = pd.read_parquet(dataset_path)
    ds = ds[ds.index >= pd.Timestamp(train_start, tz="UTC")]
    frame = pd.DataFrame(index=ds.index)
    frame["price"] = ds[PRICE_COL]
    frame["load"] = ds["load_forecast_mw"]
    frame["res"] = ds[RES_COLS].sum(axis=1, min_count=len(RES_COLS))
    # 0.6 % of neighbour hours are missing: carry the last published value forward
    frame["rl_neighbours"] = ds[NEIGHBOUR_COLS].ffill().sum(axis=1, min_count=len(NEIGHBOUR_COLS))
    for c in exog_extra:
        frame[c] = ds[c].ffill()
    frame["ttf"] = ds["ttf_gas_eur_mwh"].ffill()
    frame["eua"] = ds["eua_proxy_usd"].fillna(0.0)  # NaN before 2021-10; no back-fill

    frame = frame.dropna()
    sizes = frame.groupby(frame.index.normalize()).size()
    frame = frame[frame.index.normalize().isin(sizes[sizes == 24].index)]
    days = pd.DatetimeIndex(sorted(set(frame.index.normalize())))
    gaps = (days[1:] - days[:-1]) != pd.Timedelta(days=1)
    if gaps.any():
        # keep the last contiguous stretch
        start = days[1:][gaps][-1]
        frame = frame[frame.index >= start]
        days = days[days >= start]

    prices = frame["price"].to_numpy().reshape(-1, 24)
    exog_cols = ["load", "res", "rl_neighbours", *exog_extra]
    exog = np.stack([frame[c].to_numpy().reshape(-1, 24) for c in exog_cols], axis=2)
    fuels = frame[["ttf", "eua"]].to_numpy().reshape(-1, 24, 2).mean(axis=1)
    return prices, exog, fuels, days, frame.index


def design(prices, exog, fuels, days):
    """X rows for days[7:] (fuels inserted before the 7 unscaled dummies), Y, row days."""
    x, y = build_xy(prices, exog, np.array([d.dayofweek for d in days]), gate_safe=True)
    x = np.hstack([x[:, :-7], fuels[7:], x[:, -7:]])
    return x, y, days[7:]


def _refit(x, y, row_days, month, config, seed):
    train = row_days <= month - pd.Timedelta(days=2)
    test = (row_days >= month) & (row_days < month + pd.offsets.MonthBegin(1))
    if not test.any():
        return month, None
    import torch

    torch.set_num_threads(1)
    return month, fit_predict(x[train], y[train], x[test], 7, config, seed)


def run(
    out_dir: Path,
    first_fit: str = "2020-01-01",
    train_start: str = "2018-12-01",
    test_end: str | None = None,
    dataset_path: str = "data/dataset/hourly.parquet",
    exog_extra: list[str] | None = None,
    hidden: list[int] | None = None,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    max_epochs: int = 400,
    patience: int = 30,
    n_seeds: int = 4,
    n_jobs: int = -1,
) -> dict:
    config = QNNConfig(
        hidden=hidden or [256, 256],
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
    )
    prices, exog, fuels, days, hours = load_days(dataset_path, train_start, exog_extra or [])
    x, y, row_days = design(prices, exog, fuels, days)

    last = row_days.max() if test_end is None else pd.Timestamp(test_end, tz="UTC")
    months = pd.date_range(pd.Timestamp(first_fit, tz="UTC"), last, freq="MS")
    jobs = [(m, s) for m in months for s in range(n_seeds)]
    print(f"{len(months)} refits x {n_seeds} seeds = {len(jobs)} fits, X {x.shape}", flush=True)
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_refit)(x, y, row_days, m, config, s) for m, s in jobs
    )

    by_month: dict = {}
    for month, pred in results:
        if pred is not None:
            by_month.setdefault(month, []).append(pred)
    parts = []
    for month, preds in sorted(by_month.items()):
        q = np.sort(np.mean(preds, axis=0), axis=-1)  # (n_days, 24, 99), Vincentized
        test_days = row_days[(row_days >= month) & (row_days < month + pd.offsets.MonthBegin(1))]
        idx = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in test_days for h in range(24)])
        parts.append(pd.DataFrame(q.reshape(-1, len(QUANTILES)), index=idx, columns=Q_COLS))
    qdf = pd.concat(parts)
    if test_end is not None:
        qdf = qdf[qdf.index < pd.Timestamp(test_end, tz="UTC") + pd.Timedelta(days=1)]
    actual = pd.Series(prices.reshape(-1), index=hours).reindex(qdf.index)
    qdf["actual"] = actual
    qdf.to_parquet(out_dir / "quantiles.parquet")
    qdf[["q50", "actual"]].rename(columns={"q50": "qnn_median"}).to_parquet(
        out_dir / "forecast.parquet"
    )

    prices_all = pd.read_parquet(dataset_path)[PRICE_COL]
    metrics = {
        "params": {
            "first_fit": first_fit,
            "train_start": train_start,
            "exog_extra": exog_extra or [],
            "config": vars(config),
            "n_seeds": n_seeds,
        },
        "overall": probabilistic_metrics(qdf, prices_all),
        "by_year": {},
    }
    for year in sorted(set(qdf.index.year)):
        part = qdf[qdf.index.year == year]
        metrics["by_year"][int(year)] = probabilistic_metrics(part, prices_all)
    return metrics


def pinball_by_quantile(qdf: pd.DataFrame) -> np.ndarray:
    """Mean pinball loss per percentile, EUR/MWh."""
    diff = qdf["actual"].to_numpy()[:, None] - qdf[Q_COLS].to_numpy()
    return np.maximum(QUANTILES * diff, (QUANTILES - 1) * diff).mean(axis=0)


def probabilistic_metrics(qdf: pd.DataFrame, prices_all: pd.Series) -> dict:
    """Mean pinball over the 99 percentiles (x2 ~ CRPS), coverage, median MAE."""
    a = qdf["actual"]
    naive_w = naive_forecast(prices_all, "weekly").reindex(qdf.index)
    pb = pinball_by_quantile(qdf)
    below = qdf[Q_COLS].to_numpy() > a.to_numpy()[:, None]
    high = a > 200
    out = {
        "n_hours": len(qdf),
        "pinball_mean": round(float(pb.mean()), 3),
        "crps_approx": round(float(2 * pb.mean()), 3),
        "MAE_median": round(mae(a.values, qdf["q50"].values), 3),
        "rMAE_median": round(mae(a.values, qdf["q50"].values) / mae(a.values, naive_w.values), 4),
        "coverage_80": round(float(((a >= qdf["q10"]) & (a <= qdf["q90"])).mean()), 4),
        "coverage_98": round(float(((a >= qdf["q01"]) & (a <= qdf["q99"])).mean()), 4),
        "exceed_share": {
            c: round(float(1 - below[:, i].mean()), 4)
            for i, c in enumerate(Q_COLS)
            if c in ("q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99")
        },
    }
    if high.any():
        out["gt200"] = {
            "n": int(high.sum()),
            "pinball_mean": round(float(pinball_by_quantile(qdf[high]).mean()), 3),
            "MAE_median": round(mae(a[high].values, qdf["q50"][high].values), 3),
            "share_above_q90": round(float((a[high] > qdf["q90"][high]).mean()), 4),
            "share_above_q99": round(float((a[high] > qdf["q99"][high]).mean()), 4),
        }
    return out
