"""Equal-weight quantile mix of the nets and t0 (pandas/scipy only, no torch).

Mix = the plain average of the nets' recalibrated percentiles and t0's, level by level
(the evaluated variant: fair replay 2026-10-16, best in both editions). Averaging two
monotone quantile functions keeps the result monotone. t0 forecasts 21 levels, so it is
first put on the nets' 99-level grid. Quarter-hours keep the nets' within-hour shape.

Logged every forecast run (both editions) to `mix_log/YYYY-MM.parquet` next to the nets
log, in the nets log's layout (kind "h": 24 hourly rows, kind "qh": 96 quarter-hour rows,
final percentiles only - nothing recalibrates on the mix), with the nets run's flags plus
`components` ("nets+t0", or "nets" when t0 failed that run: then the rows are the nets'
own) and `nets_generated_utc` (the nets run the mix was built on). The nets are required:
without them nothing is logged. Published only with `pep forecast --primary mix`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pred_el_prices.production.site import (
    LOG_DIR,
    MIX_LOG_DIR,
    Q_COLS,
    QUANTILES,
    _write_partitions,
    log_exists,
    read_log,
    read_partitions,
)
from pred_el_prices.production.t0_shadow import LEVELS as T0_LEVELS
from pred_el_prices.production.t0_shadow import Q_NAMES as T0_COLS

PRICE_BOUNDS = (-500.0, 4000.0)  # SDAC limits, as experiments.qnn_de (restated: no torch)
WITH_T0, NETS_ONLY = "nets+t0", "nets"
# carried from the nets rows: provenance and substitute-input flags, as in the nets log
CARRY = [
    "trained_through",
    "weather_vintage",
    "evening",
    "load_surrogate",
    "neighbour_surrogate",
    "backfill",
]


def t0_on_grid(t0: pd.DataFrame) -> pd.DataFrame:
    """t0's 21 levels (q01, q05, q10..q90, q95, q99) on the 99-level grid (Q_COLS).

    PCHIP through the quantile function, levels -> prices. It is monotone wherever the
    knots are (no overshoot between levels, unlike a cubic spline), passes through t0's own
    levels exactly, and keeps a continuous slope, so the implied density has no steps at
    the knots as linear interpolation would give (most visible in the wide 0.01-0.05 and
    0.95-0.99 tails). The grid lies inside [0.01, 0.99], t0's own outer levels: no
    extrapolation. Knots are sorted first, as the nets' Vincentized percentiles are.
    """
    from scipy.interpolate import PchipInterpolator

    knots = np.sort(t0[T0_COLS].to_numpy(dtype=float), axis=1)
    q = PchipInterpolator(T0_LEVELS, knots, axis=1, extrapolate=False)(QUANTILES)
    return pd.DataFrame(q, index=t0.index, columns=Q_COLS)


def mix_hourly(nets_h: pd.DataFrame, t0_99: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight average of the nets' percentiles and t0's on the same grid (24 hours).

    Clipped to the SDAC limits as the nets are (t0 is not; nothing outside clears)."""
    t0_99 = t0_99.reindex(nets_h.index)
    if t0_99[Q_COLS].isna().any().any():
        raise RuntimeError("t0 rows do not cover the nets' hours")
    q = 0.5 * (nets_h[Q_COLS].to_numpy(dtype=float) + t0_99[Q_COLS].to_numpy(dtype=float))
    return pd.DataFrame(np.clip(q, *PRICE_BOUNDS), index=nets_h.index, columns=Q_COLS)


def mix_quarters(nets_qh: pd.DataFrame, nets_h: pd.DataFrame, mix_h: pd.DataFrame):
    """Quarter-hour percentiles of the mix: the nets' quarter-hours shifted, level by level,
    by what the mix moved their hour (mix_qh = nets_qh + mix_h - nets_h).

    The nets build quarter-hours as hourly percentiles + one median shape
    (qh_shape.quarter_percentiles, then clipped), so this is the mix's hourly percentiles
    + the same shape: the shape model is not refit on the mix median, which it never
    trained on. Taken per level, a level the nets' clipping touched stays consistent.
    Sorted again across levels (a no-op unless clipping bit) and clipped.
    """
    hours = nets_qh.index.floor("h")
    shift = mix_h[Q_COLS].reindex(hours).to_numpy() - nets_h[Q_COLS].reindex(hours).to_numpy()
    q = np.sort(nets_qh[Q_COLS].to_numpy(dtype=float) + shift, axis=1)
    return pd.DataFrame(np.clip(q, *PRICE_BOUNDS), index=nets_qh.index, columns=Q_COLS)


def mix_rows(nets_rows: pd.DataFrame, t0: pd.DataFrame | None, generated_utc: str):
    """One run's mix log rows from one day's standing nets rows and t0's rows (or None)."""
    h = nets_rows[nets_rows["kind"] == "h"].sort_index()
    qh = nets_rows[nets_rows["kind"] == "qh"].sort_index()
    if len(h) != 24 or len(qh) != 96:
        raise RuntimeError(f"nets rows incomplete ({len(h)} hours, {len(qh)} quarters)")
    if t0 is None:
        values_h, values_qh, components = h[Q_COLS], qh[Q_COLS], NETS_ONLY
    else:
        values_h = mix_hourly(h, t0_on_grid(t0))
        values_qh, components = mix_quarters(qh, h, values_h), WITH_T0
    values = pd.concat([values_h.astype("float32"), values_qh.astype("float32")])
    meta = pd.DataFrame(
        {
            "kind": ["h"] * 24 + ["qh"] * 96,
            "generated_utc": generated_utc,
            "nets_generated_utc": str(h["generated_utc"].iloc[0]),
            "components": components,
        },
        index=values.index,
    )
    for col in CARRY:
        if col in h:
            meta[col] = h[col].iloc[0]
    rows = pd.concat([meta, values], axis=1)
    rows.index.name = "t"
    return rows


def _evening(rows: pd.DataFrame) -> bool:
    return bool(rows.get("evening", pd.Series(False, index=rows.index)).eq(True).any())


def mix_step(
    out_dir: Path,
    delivery: pd.Timestamp,
    t0: pd.DataFrame | None,
    generated_utc: str,
    evening: bool = False,
) -> bool:
    """Mix for `delivery` from its standing nets rows and this run's t0 rows (None: t0
    failed or did not run), appended to the mix log. Never raises.

    Nothing is logged when no nets forecast of this run's edition stands: a morning run
    whose nets failed must not mix the evening nets with the morning t0.
    """
    try:
        path = Path(out_dir) / LOG_DIR
        log = read_log(path) if log_exists(path) else None
        rows = log[log.index.normalize() == delivery] if log is not None else None
        if rows is None or rows.empty or _evening(rows[rows["kind"] == "h"]) != evening:
            print(f"mix: no {'evening' if evening else 'morning'} nets for {delivery:%Y-%m-%d}")
            return False
        out = mix_rows(rows, t0, generated_utc)
        _write_partitions(Path(out_dir) / MIX_LOG_DIR, out)
        print(f"mix: {delivery:%Y-%m-%d} logged ({out['components'].iloc[0]})")
        return True
    except Exception as e:  # noqa: BLE001 - shadow path: nothing here may cost the day
        print(f"::warning::mix step failed ({type(e).__name__}: {e})")
        return False


def mix_logged(out_dir: Path, delivery: pd.Timestamp) -> bool:
    """Whether a standing morning mix WITH t0 exists for `delivery` (a retry slot's test)."""
    log = read_partitions(Path(out_dir) / MIX_LOG_DIR)
    if log is None:
        return False
    h = log[(log.index.normalize() == delivery) & (log["kind"] == "h")]
    return bool(len(h)) and not _evening(h) and bool((h["components"] == WITH_T0).all())
