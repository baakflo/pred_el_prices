"""Network forecast log and the v2 site JSON additions (pandas only, no torch).

The log is append-only, partitioned by month: `nets_log/YYYY-MM.parquet` next to the
LEAR `forecast_log.parquet` (a run rewrites only its month, so the state repo's history
grows linearly). Per delivery run it holds 24 hourly rows (kind "h": final percentiles
q01..q99 plus the raw Vincentized percentiles r01..r99 that the PIT recalibration
history needs) and 96 quarter-hour rows (kind "qh": final percentiles only), all at UTC
period starts, with the run's generated_utc and flags (backfill=True: rows from
`pep backfill-nets`, published as "replay"). The last row per (t, kind) stands, as in
the LEAR log. A single-file log (`nets_log.parquet`, the first layout; backfill runs)
is read as is; next to a partition directory it is split into partitions once and then
left alone.
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
T0_MODEL_LABEL = (
    "t0-beta foundation model (zero-shot), 21 levels on the 99-level grid, recalibrated"
)
MIX_MODEL_LABEL = "24 networks + t0-beta foundation model, equal-weight quantile mix"
# the site's model blocks (pep forecast --site-models), each one a sibling key of the
# same shape; "nets" is always among them, alone it is the site as it always was
SITE_MODELS = ("nets", "t0", "mix")
MODEL_LABELS = {"nets": NETS_MODEL_LABEL, "t0": T0_MODEL_LABEL, "mix": MIX_MODEL_LABEL}
LOG_DIR = "nets_log"  # monthly partitions
LOG_NAME = "nets_log.parquet"  # single-file layout (legacy state, backfill outputs)
MIX_LOG_DIR = "mix_log"  # nets+t0 mix (production/mix.py), monthly partitions only
MIX_WITH_T0 = "nets+t0"  # the mix log's `components` when t0 made it in
SEED_NAME = "nets_pit_seed.parquet"  # PIT/median history before go-live (pep seed-nets-pit)
T0_SEED_NAME = "t0_pit_seed.parquet"  # t0's PIT history before its log (pep seed-t0-pit)
DAYS_DIR = "days"  # per-delivery-day detail files for the site
# the 15-minute MTU went live in SDAC on 2025-10-01; before it the cache is hourly
QH_START = pd.Timestamp("2025-10-01", tz="UTC")


def _single_file(path: Path) -> bool:
    return Path(path).suffix == ".parquet"


def _write_partitions(log_dir: Path, rows: pd.DataFrame) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    for month, part in rows.groupby(rows.index.strftime("%Y-%m")):
        p = log_dir / f"{month}.parquet"
        if p.exists():
            part = pd.concat([pd.read_parquet(p), part])
        part.to_parquet(p)


def write_log(log_dir: Path, log: pd.DataFrame) -> None:
    """Write `log` as the partition set, replacing the partitions of its months."""
    log_dir.mkdir(parents=True, exist_ok=True)
    for month, part in log.groupby(log.index.strftime("%Y-%m")):
        part.to_parquet(log_dir / f"{month}.parquet")


def _migrate(log_dir: Path) -> None:
    """Split a single-file log next to `log_dir` into partitions, once; the old file stays."""
    legacy = log_dir.parent / LOG_NAME
    if not legacy.exists() or (log_dir.exists() and any(log_dir.glob("*.parquet"))):
        return
    _write_partitions(log_dir, pd.read_parquet(legacy))
    print(
        f"NOTICE: {legacy} split into monthly partitions under {log_dir}; "
        "the old file is left in place and no longer written"
    )


def log_exists(path: Path) -> bool:
    path = Path(path)
    if _single_file(path):
        return path.exists()
    return (path.exists() and any(path.glob("*.parquet"))) or (path.parent / LOG_NAME).exists()


def read_log(path: Path) -> pd.DataFrame:
    """Standing rows only: the last appended row per (t, kind).

    `path`: a partition directory (state layout) or a single parquet file.
    """
    path = Path(path)
    if _single_file(path):
        log = pd.read_parquet(path)
    else:
        _migrate(path)
        log = pd.concat([pd.read_parquet(p) for p in sorted(path.glob("*.parquet"))])
    return _standing(log)


def _standing(log: pd.DataFrame) -> pd.DataFrame:
    keep = ~pd.MultiIndex.from_arrays([log.index, log["kind"]]).duplicated(keep="last")
    return log[keep].sort_index()


def read_partitions(log_dir: Path) -> pd.DataFrame | None:
    """Standing rows of a partition directory in the nets log's layout (None when empty).

    For logs that never had a single-file past (the mix log): read_log's migration would
    split a `nets_log.parquet` lying next to them into their partitions.
    """
    parts = sorted(Path(log_dir).glob("*.parquet"))
    if not parts:
        return None
    return _standing(pd.concat([pd.read_parquet(p) for p in parts]))


def append_log(path: Path, rows: pd.DataFrame) -> None:
    path = Path(path)
    if _single_file(path):
        if path.exists():
            rows = pd.concat([pd.read_parquet(path), rows])
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(path)
        return
    _migrate(path)
    _write_partitions(path, rows)


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
    """15-minute clearing prices from the 15-min MTU start: ENTSO-E (native resolution),
    gaps filled from energy-charts' native points (same auction; it often publishes
    first — 2026-10-16: past the 12:15 refresh ENTSO-E still had nothing)."""
    from pred_el_prices.pipeline import cache
    from pred_el_prices.pipeline.energy_charts import DATASET_QH

    parts = []
    for dataset in ("entsoe/day_ahead_prices", DATASET_QH):  # first wins on overlap
        p = cache.load(cache_dir, dataset)
        if not p.empty:
            parts.append(p["price_eur_mwh"].dropna())
    if not parts:
        return pd.Series(dtype=float)
    p = pd.concat(parts)
    p = p[~p.index.duplicated(keep="first")].sort_index()
    return p[(p.index >= QH_START) & p.index.minute.isin([0, 15, 30, 45])]


def _pinball_rows(q: np.ndarray, a: np.ndarray) -> np.ndarray:
    diff = a[:, None] - q
    return np.maximum(QUANTILES * diff, (QUANTILES - 1) * diff).mean(axis=1)


def _r(x) -> float | None:
    return None if x is None or pd.isna(x) else round(float(x), 2)


# central bands scored for coverage: name -> (lower, upper) percentile column index
COVERAGE = {"cov80": (9, 89), "cov90": (4, 94), "cov98": (0, 98)}
# percentiles kept per hour in history.json (the tails show spikes inside the range)
HISTORY_LEVELS = [1, 5, 10, 50, 90, 95, 99]


def day_scores(hours: pd.DataFrame, quarters: pd.DataFrame, a_h, a_qh) -> dict:
    """mae (median), pinball (mean over 99 levels), cov80/cov90/cov98 (share inside
    [q10, q90], [q05, q95], [q01, q99]) and the _qh twins on quarter-hours; periods
    with an actual only (None where nothing is scored yet)."""
    out = {}
    for suffix, rows, a in (("", hours, a_h), ("_qh", quarters, a_qh)):
        a = pd.Series(a, index=rows.index, dtype=float)
        ok = a.notna().to_numpy()
        names = ["mae", "pinball", *COVERAGE]
        if not ok.any():
            out |= {f"{n}{suffix}": None for n in names}
            continue
        q = rows[Q_COLS].to_numpy(dtype=float)[ok]
        av = a.to_numpy()[ok]
        out[f"mae{suffix}"] = _r(np.abs(q[:, 49] - av).mean())
        out[f"pinball{suffix}"] = _r(_pinball_rows(q, av).mean())
        for name, (lo, hi) in COVERAGE.items():
            out[f"{name}{suffix}"] = _r(((av >= q[:, lo]) & (av <= q[:, hi])).mean())
    return out


def _curve(rows: pd.DataFrame, actual: pd.Series) -> list[dict]:
    a = actual.reindex(rows.index)
    return [
        {"t": t.isoformat(), "q": [_r(v) for v in vals], "actual": _r(x)}
        for t, vals, x in zip(rows.index, rows[SITE_COLS].to_numpy(), a, strict=True)
    ]


def by_day(log: pd.DataFrame) -> dict:
    """Log rows grouped per UTC delivery day (for the per-day blocks below)."""
    return dict(iter(log.groupby(log.index.normalize())))


# log flag -> published key; set only when true. "backfill" rows come from
# `pep backfill-nets` (a replay, not a live forecast); evening/surrogates mark substitute
# inputs. The last two exist on the t0/mix blocks' rows only (production/mix.site_days):
# a mix whose nets and t0 came from different editions, t0 quarters without a nets shape.
NETS_FLAGS = {
    "backfill": "replay",
    "evening": "evening",
    "load_surrogate": "load_surrogate",
    "neighbour_surrogate": "neighbour_surrogate",
    "mixed_editions": "mixed_editions",
    "flat_quarters": "flat_quarters",
}
# component runs of a mix block, published next to its own generated_utc
PROVENANCE = ("nets_generated_utc", "t0_generated_utc")


def _replay(h: pd.DataFrame) -> dict:
    """Provenance and substitute-input flags of one day's hourly rows."""
    flags = {key: True for col, key in NETS_FLAGS.items() if col in h and h[col].eq(True).any()}
    return flags | _t0_recal(h)


def _t0_recal(h: pd.DataFrame) -> dict:
    """t0's PIT recalibration on the t0/mix blocks (production/mix.T0Calibration):
    `t0_recal_days` (PIT days used) and `t0_uncalibrated` when that is 0 (too little
    history: t0's own percentiles), as the nets log's recal_days. Absent on the nets
    block and wherever no t0 was recalibrated."""
    days = h["t0_recal_days"].dropna() if "t0_recal_days" in h else pd.Series(dtype=float)
    if days.empty:
        return {}
    n = int(days.iloc[0])
    return {"t0_recal_days": n, **({"t0_uncalibrated": True} if n == 0 else {})}


def latest_nets(rows: pd.DataFrame, prices, prices_qh, model: str = "nets") -> dict | None:
    """A model block of latest.json from one day's rows in the nets log layout (None
    without hourly rows): `nets` from the nets log, `t0`/`mix` from mix.site_days. t0 is
    zero-shot and has no trained_through."""
    h, qh = rows[rows["kind"] == "h"], rows[rows["kind"] == "qh"]
    if h.empty:
        return None
    trained = h.get("trained_through")  # t0 is zero-shot: none
    return {
        "model": MODEL_LABELS[model],
        **({"trained_through": str(trained.iloc[0])} if trained is not None else {}),
        "generated_utc": str(h["generated_utc"].iloc[0]),
        **{col: str(h[col].iloc[0]) for col in PROVENANCE if col in h},
        **_replay(h),
        "hours": _curve(h, prices),
        "quarters": _curve(qh, prices_qh),
    }


def day_nets(rows: pd.DataFrame, prices, prices_qh, model: str = "nets") -> dict | None:
    """A model block of a day file: latest.json's block plus the day's scores."""
    block = latest_nets(rows, prices, prices_qh, model)
    if block is None:
        return None
    h, qh = rows[rows["kind"] == "h"], rows[rows["kind"] == "qh"]
    return {**block, **day_scores(h, qh, prices.reindex(h.index), prices_qh.reindex(qh.index))}


def history_nets(rows: pd.DataFrame, prices, prices_qh) -> dict | None:
    """A model block of a history.json day: scores plus the hourly curve at
    HISTORY_LEVELS (keys q1, q5, q10, q50, q90, q95, q99)."""
    h, qh = rows[rows["kind"] == "h"], rows[rows["kind"] == "qh"]
    if h.empty:
        return None
    a_h, a_qh = prices.reindex(h.index), prices_qh.reindex(qh.index)
    cols = [f"q{lv:02d}" for lv in HISTORY_LEVELS]
    return {
        **day_scores(h, qh, a_h, a_qh),
        **_replay(h),
        "hours": [
            {
                "t": t.isoformat(),
                **{f"q{lv}": _r(v) for lv, v in zip(HISTORY_LEVELS, vals, strict=True)},
                "actual": _r(x),
            }
            for t, vals, x in zip(h.index, h[cols].to_numpy(), a_h, strict=True)
        ],
    }
