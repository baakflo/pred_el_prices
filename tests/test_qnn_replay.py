"""Leakage tests for the production replay (offline, synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from pred_el_prices.experiments import qnn_replay
from pred_el_prices.models.qnn import QUANTILES


def _days(n: int):
    rng = np.random.default_rng(0)
    days = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    prices = 50 + rng.normal(0, 10, (n, 24))
    exog = rng.normal(40000, 3000, (n, 24, 3))
    fuels = np.column_stack([30 + rng.normal(0, 1, n), 70 + rng.normal(0, 1, n)])
    hours = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in days for h in range(24)])
    return prices, exog, fuels, days, hours


def _run(monkeypatch, tmp_path, exog_edit=None):
    prices, exog, fuels, days, hours = _days(60)
    if exog_edit:
        exog_edit(exog, days)
    monkeypatch.setattr(qnn_replay, "load_days", lambda *a, **k: (prices, exog, fuels, days, hours))
    seen = []

    def spy(x_train, y_train, x_pred, n_unscaled, config, seed):
        seen.append(x_pred.copy())
        return np.zeros((len(x_pred), 24, len(QUANTILES)))

    monkeypatch.setattr(qnn_replay, "fit_predict", spy)
    own = pd.Series(
        20000.0, index=pd.date_range("2024-02-20", periods=24 * 10, freq="1h", tz="UTC")
    )
    own.rename("own_res_mw").to_frame().to_parquet(tmp_path / "own.parquet")
    qnn_replay.run(
        tmp_path, "2024-02-20", "2024-02-26", str(tmp_path / "own.parquet"),
        n_seeds=1, heads=["quantile"], n_jobs=1,
    )  # fmt: skip
    x = seen[0]
    return x[: len(x) // 2], x[len(x) // 2 :]  # prod rows, tso rows


def test_prod_rows_ignore_the_post_gate_tso_res_and_boundary_load(monkeypatch, tmp_path):
    prod_a, tso_a = _run(monkeypatch, tmp_path)

    def bump(exog, days):
        k = days.get_loc(pd.Timestamp("2024-02-22", tz="UTC"))
        exog[k, :, 1] += 5000.0  # TSO RES of the delivery day: published after the gate
        exog[k, 22:24, 0] += 5000.0  # DE load of the next local day
        exog[k, 22:24, 2] += 5000.0  # neighbour load of the next local day

    prod_b, tso_b = _run(monkeypatch, tmp_path, bump)
    row = 2  # 2024-02-22 is the third day of the window
    np.testing.assert_array_equal(prod_a[row], prod_b[row])
    assert not np.array_equal(tso_a[row], tso_b[row])


def test_prod_rows_keep_tso_lags(monkeypatch, tmp_path):
    prod, tso = _run(monkeypatch, tmp_path)
    # the row for D+1 carries D's TSO RES as its d-1 lag, which production has
    # (published 18:00 on D-1): prod and tso rows may differ only in D+1's own lag-0
    # RES block, the fuels (identical) and the boundary-filled load hours
    diff_cols = np.where(prod[3] != tso[3])[0]
    assert len(diff_cols) <= 24 + 2 + 2
