"""Step-in for gaps in the DE_LU TSO day-ahead forecasts from the four control areas.

2026-10-24: TransnetBW never submitted its day-ahead wind/solar forecast, so the
DE_LU aggregate on ENTSO-E (and SMARD, which mirrors it) had NaN for onshore and
solar over the whole local day while 50Hertz, Amprion and TenneT had reported.
This module reconstructs such a DE_LU value from the areas that did report:

    DE_LU(t) = sum_{a in R} x_a(t) / s_R(t),   s_R = sum_{a in R} x_a / DE_LU

with R the reporting set at t and s_R the share of R in the aggregate over the
trailing `WINDOW_DAYS` days strictly before t's UTC day on which DE_LU and every
area of R are complete (per hour of day where `PER_HOUR`, else pooled). The share
absorbs Luxembourg and any area-sum != aggregate offset.

Rules: it only steps in where the DE_LU value is missing after the caller's
ENTSO-E -> SMARD ladder AND at least one area reported; a ladder value is never
touched; no area reported -> no fill; a failed guard (fewer than `MIN_DAYS`
complete days, share not finite or outside (SHARE_FLOOR, SHARE_CEIL)) leaves NaN.
Night solar with no defined share (0/0) is 0 only if the reporting areas say 0
and DE_LU at that hour was ~0 over the whole window.

Leakage: the area series are the same A65 / A69 day-ahead submissions (process
A01) the aggregate is built from, published at the same time; the share uses only
days before t's day. Intraday/current forecasts are never read.

Validation (normal days 2025-11..2026-10, every proper subset of the areas
hidden and rebuilt): per-hour shares win for solar (6.2 vs 6.6 % mean MAE over the
subsets) and load (2.6 vs 2.7 %), wind is indifferent (pooled kept); windows of
14/28/56 days are within 0.1-0.4 pp, 28 kept. Sum of areas / DE_LU on complete
days: load 0.990 (LU ~560 MW), onshore 0.996, offshore 1.000, solar 0.991.
Per-subset errors and the wind cap: see MIN_REPORTED_SHARE.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pred_el_prices.pipeline import cache

# entsoe-py area codes of the four German TSOs (control areas)
CONTROL_AREAS = ("DE_50HZ", "DE_AMPRION", "DE_TENNET", "DE_TRANSNET")
AREA_DATASETS = ("load_forecast", "wind_solar_forecast")

# DE_LU output column -> (entsoe dataset, entsoe column)
AREA_COLUMNS = {
    "load_forecast_mw": ("load_forecast", "Forecasted Load"),
    "wind_onshore_forecast_mw": ("wind_solar_forecast", "Wind Onshore"),
    "wind_offshore_forecast_mw": ("wind_solar_forecast", "Wind Offshore"),
    "solar_forecast_mw": ("wind_solar_forecast", "Solar"),
}
ENTSOE_TO_COLUMN = {e_col: col for col, (_, e_col) in AREA_COLUMNS.items()}

WINDOW_DAYS = 28  # trailing calendar days the share is measured over
MIN_DAYS = 14  # complete days needed inside the window, else no fill
# per hour of day (diurnal shape differs by area) vs one pooled share
PER_HOUR = {
    "load_forecast_mw": True,
    "wind_onshore_forecast_mw": False,
    "wind_offshore_forecast_mw": False,
    "solar_forecast_mw": True,
}
# Cap: no fill when the reporting areas carried less than this share of the
# aggregate over the window (pooled). Hold-out 2025-11..2026-10, MAE as % of the
# mean, against the TSO forecast's own day-ahead error vs actuals (onshore 9.9 %,
# offshore 17.7 %): onshore fills stay <= 15 % while the missing areas' share is
# <= 0.55 (TransnetBW alone 1.7 %, Amprion 8.3 %, TenneT 12.2 %), but jump to
# 23-56 % beyond 0.65 (Amprion+TenneT, 50Hertz+TenneT, all but TransnetBW);
# offshore without TenneT (82 %) is 48 %. Those would more than double the input's
# own error, so the day is left out instead (a training day costs 1 of ~1400;
# LEAR's calibration has the own-RES fallback). Solar (<= 14 %) and load (<= 6 %)
# stay uncapped: every subset fills.
MIN_REPORTED_SHARE = {"wind_onshore_forecast_mw": 0.4, "wind_offshore_forecast_mw": 0.4}
SHARE_FLOOR = 0.02
SHARE_CEIL = 1.5  # the areas never exceed the aggregate by half (LU is ~1 %)
NIGHT_MW = 1.0  # "zero" for the night-solar rule
RECENT_DAYS = 8  # fills this close to the data's end raise a workflow annotation


def area_frame(cache_root: Path, column: str, freq: str = "1h") -> pd.DataFrame:
    """One column per control area that reports `column`, at `freq` (mean)."""
    dataset, e_col = AREA_COLUMNS[column]
    parts = {}
    for area in CONTROL_AREAS:
        df = cache.load(cache_root, f"entsoe/{area}/{dataset}")
        if df.empty or e_col not in df.columns:
            continue
        s = df[e_col].resample(freq).mean()
        if s.notna().any():
            parts[area] = s
    return pd.DataFrame(parts)


def _steps_per_day(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 24
    step = pd.Series(index).diff().dropna().min()
    return int(pd.Timedelta(days=1) / step)


def fill_from_areas(
    target: pd.Series,
    areas: pd.DataFrame,
    per_hour: bool = True,
    window_days: int = WINDOW_DAYS,
    min_days: int = MIN_DAYS,
    min_reported_share: float = 0.0,
) -> tuple[pd.Series, dict]:
    """`target` with its NaN filled from `areas` where the rules allow; plus counts.

    `target` and `areas` share one regular UTC index (hourly or 15-min); areas
    absent from a timestamp are NaN. Returns (filled series, info) with info =
    {"filled": rows filled, "guarded": candidate rows left NaN by a guard,
     "capped": candidate rows left NaN because the reporting areas' pooled share
     was below `min_reported_share`,
     "missing_sets": {"DE_X+DE_Y": rows filled with that set missing}}.
    """
    info: dict = {"filled": 0, "guarded": 0, "capped": 0, "missing_sets": {}}
    out = target.copy()
    if areas.empty:
        return out, info
    areas = areas.reindex(target.index)
    present = areas.notna()
    cand = target.isna().to_numpy() & present.any(axis=1).to_numpy()
    if not cand.any():
        return out, info

    idx = target.index
    day = idx.normalize()
    hod = idx.hour
    per_day = _steps_per_day(idx)
    orig_ok = target.notna().to_numpy()
    vals = areas.to_numpy(dtype=float)
    tgt = target.to_numpy(dtype=float)
    cols = list(areas.columns)
    filled = out.to_numpy(dtype=float).copy()

    # reporting set of every candidate row, as a tuple of column positions
    cand_pos = np.flatnonzero(cand)
    rsets = [tuple(np.flatnonzero(present.to_numpy()[i])) for i in cand_pos]
    groups: dict[tuple, list[int]] = {}
    for i, r in zip(cand_pos, rsets, strict=True):
        groups.setdefault((day[i], r), []).append(i)

    for (d, r), rows in groups.items():
        r = list(r)
        win = (day >= d - pd.Timedelta(days=window_days)) & (day < d)
        w = np.flatnonzero(win)
        ok = orig_ok[w] & ~np.isnan(vals[w][:, r]).any(axis=1)
        counts = pd.Series(ok).groupby(day[w]).sum()
        complete = counts.index[counts.to_numpy() == per_day]
        use = np.zeros(len(idx), dtype=bool)
        use[w] = ok & day[w].isin(complete)
        if len(complete) < min_days:
            info["guarded"] += len(rows)
            continue
        num = vals[use][:, r].sum(axis=1)
        den = tgt[use]
        if min_reported_share > 0 and not num.sum() >= min_reported_share * den.sum():
            info["capped"] += len(rows)
            continue
        if per_hour:
            g_num = pd.Series(num).groupby(hod[use]).sum()
            g_den = pd.Series(den).groupby(hod[use]).sum()
            g_n = pd.Series(den).groupby(hod[use]).size()
        missing = "+".join(cols[j] for j in range(len(cols)) if j not in r) or "none"
        n_ok = 0
        for i in rows:
            x = vals[i, r].sum()
            if per_hour:
                h = hod[i]
                if h not in g_den.index:
                    info["guarded"] += 1
                    continue
                sn, sd, n_h = g_num[h], g_den[h], g_n[h]
            else:
                sn, sd, n_h = num.sum(), den.sum(), len(den)
            share = sn / sd if sd > 0 else np.nan
            if np.isfinite(share) and SHARE_FLOOR < share < SHARE_CEIL:
                filled[i] = x / share
            elif sd / max(n_h, 1) <= NIGHT_MW and abs(x) <= NIGHT_MW:
                filled[i] = 0.0  # night: the aggregate was ~0 all window, R says 0
            else:
                info["guarded"] += 1
                continue
            n_ok += 1
        if n_ok:
            info["filled"] += n_ok
            info["missing_sets"][missing] = info["missing_sets"].get(missing, 0) + n_ok
    return pd.Series(filled, index=idx, name=target.name), info


def entsoe_de_series(
    cache_root: Path, column: str, freq: str = "1h", start: pd.Timestamp | None = None
) -> pd.Series:
    """The ENTSO-E DE_LU aggregate for `column` at `freq`, gaps stepped in from the areas.

    For the readers that use ENTSO-E alone (no SMARD rung): the live load input,
    the load surrogate's history, the networks' gate-time load, the 15-min shape.
    """
    dataset, e_col = AREA_COLUMNS[column]
    df = cache.load(cache_root, f"entsoe/{dataset}", start=start)
    if df.empty or e_col not in df.columns:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"), name=e_col)
    s = df[e_col].resample(freq).mean()
    filled, _ = fill_de_forecast(cache_root, column, s, freq, label=f"entsoe {freq}")
    return filled.rename(e_col)


def fill_de_forecast(
    cache_root: Path,
    column: str,
    series: pd.Series,
    freq: str = "1h",
    label: str = "",
) -> tuple[pd.Series, dict]:
    """Fill gaps in the DE_LU `column` series (already through the caller's ladder).

    The index is extended by every timestamp an area reported, so a DE_LU row that
    is absent (not just NaN) counts as missing too. Prints one warning line per fill.
    """
    areas = area_frame(cache_root, column, freq)
    if areas.empty:
        return series, {"filled": 0, "guarded": 0, "capped": 0, "missing_sets": {}}
    reported = areas.index[areas.notna().any(axis=1).to_numpy()]
    idx = series.index.union(reported[reported >= series.index.min()])
    filled, info = fill_from_areas(
        series.reindex(idx),
        areas,
        PER_HOUR[column],
        min_reported_share=MIN_REPORTED_SHARE.get(column, 0.0),
    )
    if info["capped"]:
        print(
            f"WARN: DE_LU {column}: {info['capped']} row(s) not filled from control areas "
            f"(reporting areas below {MIN_REPORTED_SHARE[column]:.0%} of the aggregate)"
        )
    if info["filled"]:
        unit = "h" if freq == "1h" else f"x {freq}"
        new = filled.notna() & series.reindex(idx).isna()
        span = f"{new[new].index.min():%Y-%m-%d %H:%M}..{new[new].index.max():%Y-%m-%d %H:%M}"
        sets = ", ".join(f"{k}: {v}" for k, v in info["missing_sets"].items())
        where = f" [{label}]" if label else ""
        # a GitHub annotation only while the fill is recent; the dataset is rebuilt
        # from full history every run, an old step-in stays a plain log line
        recent = new[new].index.max() >= idx.max() - pd.Timedelta(days=RECENT_DAYS)
        print(
            f"{'::warning::' if recent else 'note: '}DE_LU {column} filled from control "
            f"areas for {info['filled']} {unit} {span} UTC (missing: {sets}){where}"
        )
    return filled, info
