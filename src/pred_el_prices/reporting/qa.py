"""Data-QA checks over cached ENTSO-E frames.

All functions take UTC-indexed frames (native resolution) and return plain
dicts/lists so results can be dumped as JSON run artifacts and rendered into
report pages.
"""

import pandas as pd


def coverage_by_year(df: pd.DataFrame) -> list[dict]:
    """Hourly coverage per year: hours with at least one value vs hours in span."""
    if df.empty:
        return []
    hourly = df.resample("1h").count().sum(axis=1) > 0
    out = []
    for year, present in hourly.groupby(hourly.index.year):
        out.append(
            {
                "year": int(year),
                "hours_present": int(present.sum()),
                "hours_in_span": len(present),
                "coverage_pct": round(100 * present.sum() / len(present), 3),
            }
        )
    return out


def missing_hours(df: pd.DataFrame, max_list: int = 200) -> dict:
    """Hours inside the data span with no value at all."""
    if df.empty:
        return {"count": 0, "hours": []}
    hourly = df.resample("1h").count().sum(axis=1)
    gaps = hourly.index[hourly == 0]
    return {
        "count": len(gaps),
        "hours": [str(t) for t in gaps[:max_list]],
        "truncated": len(gaps) > max_list,
    }


def duplicate_timestamps(df: pd.DataFrame) -> int:
    return int(df.index.duplicated().sum())


def resolution_switches(df: pd.DataFrame) -> list[dict]:
    """Days where the dominant index spacing changes (e.g. hourly -> 15-min in 2025)."""
    if len(df) < 3:
        return []
    spacing = pd.Series(df.index[1:] - df.index[:-1], index=df.index[1:])
    daily_mode = spacing.groupby(spacing.index.normalize()).agg(
        lambda s: s.mode().iloc[0] if len(s) else pd.NaT
    )
    switches = []
    prev = None
    for day, mode in daily_mode.items():
        if prev is not None and mode != prev:
            switches.append({"date": str(day.date()), "from": str(prev), "to": str(mode)})
        prev = mode
    return switches


def price_stats_by_year(prices: pd.Series) -> list[dict]:
    """Yearly distribution stats for the day-ahead price (native resolution)."""
    out = []
    for year, p in prices.dropna().groupby(prices.dropna().index.year):
        out.append(
            {
                "year": int(year),
                "n": len(p),
                "min": round(float(p.min()), 2),
                "mean": round(float(p.mean()), 2),
                "max": round(float(p.max()), 2),
                "std": round(float(p.std()), 2),
                "n_negative": int((p < 0).sum()),
                "n_above_500": int((p > 500).sum()),
            }
        )
    return out


def dst_day_check(df: pd.DataFrame, years: list[int]) -> list[dict]:
    """Row counts on DST-transition days in Europe/Berlin local time.

    A correct hourly dataset has 23 rows on the spring-forward day and 25 on
    the fall-back day (46/50 at 15-min resolution).
    """
    if df.empty:
        return []
    local = df.tz_convert("Europe/Berlin")
    counts = local.groupby(local.index.normalize()).size()
    out = []
    for day, n in counts.items():
        if day.year in years and _is_dst_transition_day(day):
            out.append({"date": str(day.date()), "rows": int(n)})
    return out


def cross_check(a: pd.Series, b: pd.Series, tolerance: float = 0.01) -> dict:
    """Agreement between two hourly views of the same quantity (e.g. two portals)."""
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    diff = (joined["a"] - joined["b"]).abs()
    return {
        "n_common_hours": len(joined),
        "mean_abs_diff": round(float(diff.mean()), 4),
        "max_abs_diff": round(float(diff.max()), 4),
        "n_beyond_tolerance": int((diff > tolerance).sum()),
        "correlation": round(float(joined["a"].corr(joined["b"])), 6),
    }


def _is_dst_transition_day(day: pd.Timestamp) -> bool:
    # a naive +1 day on a tz-aware timestamp adds 24 absolute hours, which is
    # exactly wrong on transition days; go via wall-clock midnight instead
    next_midnight = (day.tz_localize(None) + pd.Timedelta(days=1)).tz_localize(day.tz)
    n_hours = (next_midnight - day) / pd.Timedelta(hours=1)
    return n_hours != 24
