"""Own pre-gate RES forecast from the archived ECMWF ENS runs (`res-de`).

No public source publishes the German day-ahead wind/solar forecasts before
the 12:00 CEST auction gate (see the 2026-08-15 plan addendum), so the live
daily price forecast must bring its own. Targets are the TSO day-ahead
forecast series — that is what LEAR trained on and what the market prices
off — normalized to capacity factors so fleet growth lives in an explicit
multiplier instead of inside a tree model that cannot extrapolate.

Evaluation is expanding-window with monthly refits: each calendar month is
predicted by a model trained strictly on earlier data, mirroring production.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from pred_el_prices.pipeline.capacity import hourly_capacity

TARGETS = {
    "wind_onshore_forecast_mw": "wind_onshore_capacity_mw",
    "wind_offshore_forecast_mw": "wind_offshore_capacity_mw",
    "solar_forecast_mw": "solar_capacity_mw",
}


def _design_matrix(features: pd.DataFrame) -> pd.DataFrame:
    x = features.drop(columns=["run_date"])
    x["hour"] = features.index.hour
    x["doy"] = features.index.dayofyear
    return x


def run(
    out_dir: Path,
    features_path: str = "data/dataset/ens_features.parquet",
    dataset_path: str = "data/dataset/hourly.parquet",
    cache_dir: str = "data/cache",
    first_fit: str = "2024-10-01",
) -> dict:
    features = pd.read_parquet(features_path)
    targets = pd.read_parquet(dataset_path)[list(TARGETS)]
    joined_index = features.index.intersection(targets.dropna().index)
    features = features.loc[joined_index]
    targets = targets.loc[joined_index]
    capacity = hourly_capacity(Path(cache_dir), joined_index)

    x = _design_matrix(features)
    cf = pd.DataFrame({t: targets[t] / capacity[cap_col].values for t, cap_col in TARGETS.items()})

    month_starts = pd.date_range(pd.Timestamp(first_fit, tz="UTC"), joined_index.max(), freq="MS")
    preds = {}
    for target in TARGETS:
        parts = []
        for month in month_starts:
            train = joined_index < month
            test = (joined_index >= month) & (joined_index < month + pd.offsets.MonthBegin(1))
            if not test.any():
                continue
            model = HistGradientBoostingRegressor(random_state=0)
            model.fit(x[train], cf[target][train])
            parts.append(pd.Series(model.predict(x[test]), index=joined_index[test]))
        cf_pred = pd.concat(parts).clip(lower=0.0)
        preds[target] = cf_pred * capacity[TARGETS[target]].reindex(cf_pred.index).values
        print(f"{target}: {len(cf_pred)} OOS hours predicted", flush=True)

    pred_df = pd.DataFrame(preds)
    eval_index = pred_df.index
    metrics: dict = {"n_hours": len(eval_index), "n_days": int(eval_index.normalize().nunique())}
    for target in TARGETS:
        err = (pred_df[target] - targets[target].reindex(eval_index)).abs()
        cap_mean = float(capacity[TARGETS[target]].reindex(eval_index).mean())
        ss_res = float(((pred_df[target] - targets[target].reindex(eval_index)) ** 2).sum())
        ss_tot = float(
            (
                (targets[target].reindex(eval_index) - targets[target].reindex(eval_index).mean())
                ** 2
            ).sum()
        )
        metrics[target] = {
            "MAE_mw": round(float(err.mean()), 1),
            "nMAE_pct_capacity": round(100 * float(err.mean()) / cap_mean, 2),
            "R2": round(1 - ss_res / ss_tot, 4),
        }
    agg_err = (pred_df.sum(axis=1) - targets.reindex(eval_index).sum(axis=1)).abs()
    metrics["aggregate_MAE_mw"] = round(float(agg_err.mean()), 1)

    out = pred_df.rename(columns={t: f"own_{t}" for t in TARGETS})
    out["own_res_forecast_mw"] = pred_df.sum(axis=1)
    out.to_parquet(out_dir / "predictions.parquet")
    return metrics
