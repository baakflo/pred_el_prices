"""Scorecard for a single forecast run, and A/B comparison between two runs."""

from pathlib import Path

import pandas as pd

from pred_el_prices.eval.metrics import (
    dm_test,
    mae,
    mae_by_hour,
    naive_forecast,
    negative_hour_metrics,
    rmse,
)


def load_forecast(run_dir: Path) -> pd.DataFrame:
    """Load runs/<name>/forecast.parquet as a DataFrame with columns `pred`, `actual`."""
    fc = pd.read_parquet(Path(run_dir) / "forecast.parquet")
    pred_cols = [c for c in fc.columns if c != "actual"]
    if "actual" not in fc.columns or len(pred_cols) != 1:
        raise ValueError(
            f"expected one prediction column and 'actual' in {run_dir}, got {list(fc.columns)}"
        )
    return fc.rename(columns={pred_cols[0]: "pred"})[["pred", "actual"]]


def score(fc: pd.DataFrame, prices_all: pd.Series) -> dict:
    """Scorecard for a single forecast frame (columns `pred`, `actual`)."""
    actual, pred = fc["actual"], fc["pred"]
    naive_w = naive_forecast(prices_all, "weekly").reindex(fc.index)
    hour = fc.index.hour
    h16_18 = actual[hour.isin([16, 17, 18])], pred[hour.isin([16, 17, 18])]
    h00_06 = actual[hour.isin(range(7))], pred[hour.isin(range(7))]
    return {
        "n_hours": len(fc),
        "MAE": round(mae(actual.values, pred.values), 3),
        "RMSE": round(rmse(actual.values, pred.values), 3),
        "rMAE_weekly": round(
            mae(actual.values, pred.values) / mae(actual.values, naive_w.values), 3
        ),
        "MAE_h16_18": round(mae(h16_18[0].values, h16_18[1].values), 3),
        "MAE_h00_06": round(mae(h00_06[0].values, h00_06[1].values), 3),
        "mae_by_hour": {h: round(v, 3) for h, v in mae_by_hour(actual, pred).items()},
        "negative": {
            k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in negative_hour_metrics(actual, pred).items()
        },
    }


def _common_whole_days(fc_a: pd.DataFrame, fc_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict two forecast frames to their common index, whole UTC days only."""
    common = fc_a.index.intersection(fc_b.index).sort_values()
    day = common.normalize()
    full_days = day.value_counts()
    full_days = full_days[full_days == 24].index
    common = common[day.isin(full_days)]
    return fc_a.loc[common], fc_b.loc[common]


def compare(fc_a: pd.DataFrame, fc_b: pd.DataFrame) -> dict:
    """Compare two forecast frames (A = baseline, B = candidate) on their common index."""
    a, b = _common_whole_days(fc_a, fc_b)
    actual = a["actual"]
    mae_a = mae(actual.values, a["pred"].values)
    mae_b = mae(actual.values, b["pred"].values)
    dm_b_better, p_b_better = dm_test(actual, a["pred"], b["pred"])
    dm_a_better, p_a_better = dm_test(actual, b["pred"], a["pred"])
    err_a = (actual - a["pred"]).abs().to_numpy().reshape(-1, 24).sum(axis=1)
    err_b = (actual - b["pred"]).abs().to_numpy().reshape(-1, 24).sum(axis=1)
    return {
        "n_days": len(err_a),
        "MAE_a": round(mae_a, 3),
        "MAE_b": round(mae_b, 3),
        "delta": round(mae_b - mae_a, 3),
        "dm_b_better_than_a": {"stat": round(dm_b_better, 3), "p": round(p_b_better, 4)},
        "dm_a_better_than_b": {"stat": round(dm_a_better, 3), "p": round(p_a_better, 4)},
        "share_days_b_wins": round(float((err_b < err_a).mean()), 3),
    }
