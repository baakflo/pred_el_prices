"""Network forecast log and the v2 site JSON additions (pandas only, no torch).

The log (`nets_log.parquet` next to the LEAR `forecast_log.parquet`) is append-only.
Per delivery run it holds 24 hourly rows (kind "h": final percentiles q01..q99 plus the
raw Vincentized percentiles r01..r99 that the PIT recalibration history needs) and 96
quarter-hour rows (kind "qh": final percentiles only), all at UTC period starts, with
the run's generated_utc and flags. The last row per (t, kind) stands, as in the LEAR log.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# same levels/columns as models.qnn / experiments.qnn_de, restated so that the site JSON
# path never imports torch (the refresh slots and LEAR must run without it)
QUANTILES = np.arange(1, 100) / 100.0
Q_COLS = [f"q{round(q * 100):02d}" for q in QUANTILES]
R_COLS = [c.replace("q", "r") for c in Q_COLS]
SITE_LEVELS = [1, 5, 10, 25, 50, 75, 90, 95, 99]
SITE_COLS = [f"q{lv:02d}" for lv in SITE_LEVELS]
NETS_MODEL_LABEL = "24 networks (12 JSU + 12 quantile), recalibrated"
LOG_NAME = "nets_log.parquet"
SEED_NAME = "nets_pit_seed.parquet"  # PIT/median history before go-live (pep seed-nets-pit)
# the 15-minute MTU went live in SDAC on 2025-10-01; before it the cache is hourly
QH_START = pd.Timestamp("2025-10-01", tz="UTC")


def read_log(path: Path) -> pd.DataFrame:
    """Standing rows only: the last appended row per (t, kind)."""
    log = pd.read_parquet(path)
    keep = ~pd.MultiIndex.from_arrays([log.index, log["kind"]]).duplicated(keep="last")
    return log[keep].sort_index()


def append_log(path: Path, rows: pd.DataFrame) -> None:
    if path.exists():
        rows = pd.concat([pd.read_parquet(path), rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(path)


def log_rows(
    hourly: pd.DataFrame, quarters: pd.DataFrame, flags: dict, generated_utc: str
) -> pd.DataFrame:
    """One run's rows: hourly (Q_COLS + R_COLS) and quarter-hour (Q_COLS) percentiles."""
    values = pd.concat(
        [hourly[Q_COLS + R_COLS].astype("float32"), quarters[Q_COLS].astype("float32")]
    )
    meta = pd.DataFrame(
        {"kind": ["h"] * len(hourly) + ["qh"] * len(quarters), "generated_utc": generated_utc},
        index=values.index,
    ).assign(**flags)
    rows = pd.concat([meta, values], axis=1)
    rows.index.name = "t"
    return rows


def quarter_prices(cache_dir: Path) -> pd.Series:
    """15-minute clearing prices (ENTSO-E native resolution) from the 15-min MTU start."""
    from pred_el_prices.pipeline import cache

    p = cache.load(cache_dir, "entsoe/day_ahead_prices")
    if p.empty:
        return pd.Series(dtype=float)
    p = p["price_eur_mwh"].dropna()
    return p[(p.index >= QH_START) & p.index.minute.isin([0, 15, 30, 45])]


def _pinball_rows(q: np.ndarray, a: np.ndarray) -> np.ndarray:
    diff = a[:, None] - q
    return np.maximum(QUANTILES * diff, (QUANTILES - 1) * diff).mean(axis=1)


def _r(x) -> float | None:
    return None if x is None or pd.isna(x) else round(float(x), 2)


def day_scores(hours: pd.DataFrame, quarters: pd.DataFrame, a_h, a_qh) -> dict:
    """mae (median), pinball (mean over 99 levels), cov80, and the _qh twins; hours with
    an actual only (None where no hour is scored yet)."""
    out = {}
    for suffix, rows, a in (("", hours, a_h), ("_qh", quarters, a_qh)):
        a = pd.Series(a, index=rows.index, dtype=float)
        ok = a.notna().to_numpy()
        if not ok.any():
            out |= {f"mae{suffix}": None, f"pinball{suffix}": None}
            if not suffix:
                out["cov80"] = None
            continue
        q = rows[Q_COLS].to_numpy(dtype=float)[ok]
        av = a.to_numpy()[ok]
        out[f"mae{suffix}"] = _r(np.abs(q[:, 49] - av).mean())
        out[f"pinball{suffix}"] = _r(_pinball_rows(q, av).mean())
        if not suffix:
            out["cov80"] = _r(((av >= q[:, 9]) & (av <= q[:, 89])).mean())
    return out


def _curve(rows: pd.DataFrame, actual: pd.Series) -> list[dict]:
    a = actual.reindex(rows.index)
    return [
        {"t": t.isoformat(), "q": [_r(v) for v in vals], "actual": _r(x)}
        for t, vals, x in zip(rows.index, rows[SITE_COLS].to_numpy(), a, strict=True)
    ]


def latest_nets(log: pd.DataFrame, day: pd.Timestamp, prices, prices_qh) -> dict | None:
    """The `nets` block of latest.json for delivery `day`, or None if the nets have none."""
    rows = log[log.index.normalize() == day]
    h, qh = rows[rows["kind"] == "h"], rows[rows["kind"] == "qh"]
    if h.empty:
        return None
    return {
        "model": NETS_MODEL_LABEL,
        "trained_through": str(h["trained_through"].iloc[0]),
        "generated_utc": str(h["generated_utc"].iloc[0]),
        "hours": _curve(h, prices),
        "quarters": _curve(qh, prices_qh),
    }


def history_nets(log: pd.DataFrame, day: pd.Timestamp, prices, prices_qh) -> dict | None:
    """The `nets` block of a history.json day: scores plus the q10/q50/q90 hourly curve."""
    rows = log[log.index.normalize() == day]
    h, qh = rows[rows["kind"] == "h"], rows[rows["kind"] == "qh"]
    if h.empty:
        return None
    a_h, a_qh = prices.reindex(h.index), prices_qh.reindex(qh.index)
    return {
        **day_scores(h, qh, a_h, a_qh),
        "hours": [
            {"t": t.isoformat(), "q10": _r(q10), "q50": _r(q50), "q90": _r(q90), "actual": _r(x)}
            for t, q10, q50, q90, x in zip(h.index, h["q10"], h["q50"], h["q90"], a_h, strict=True)
        ],
    }
