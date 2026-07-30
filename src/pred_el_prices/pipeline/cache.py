"""Local Parquet cache for time-series datasets.

Layout: <root>/<dataset>/<year>.parquet, one wide DataFrame per file with a
UTC DatetimeIndex. Writes are idempotent upserts: new rows win on overlapping
timestamps, columns are unioned (missing values become NaN).
"""

from pathlib import Path

import pandas as pd


def _dataset_dir(root: Path, dataset: str) -> Path:
    return root / dataset


def _to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        raise ValueError("cache requires a tz-aware DatetimeIndex")
    df = df.copy()
    df.index = df.index.tz_convert("UTC")
    df.index.name = "time_utc"
    return df.sort_index()


def upsert(root: Path, dataset: str, df: pd.DataFrame) -> None:
    """Merge rows into the per-year files; on duplicate timestamps the new data wins."""
    if df.empty:
        return
    df = _to_utc_index(df)
    out_dir = _dataset_dir(root, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    for year, chunk in df.groupby(df.index.year):
        path = out_dir / f"{year}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            chunk = pd.concat([existing, chunk])
            chunk = chunk[~chunk.index.duplicated(keep="last")].sort_index()
        chunk.to_parquet(path)


def load(
    root: Path,
    dataset: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Concatenate all cached years of a dataset, optionally sliced to [start, end)."""
    files = sorted(_dataset_dir(root, dataset).glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    if start is not None:
        df = df[df.index >= start.tz_convert("UTC")]
    if end is not None:
        df = df[df.index < end.tz_convert("UTC")]
    return df


def last_timestamp(root: Path, dataset: str) -> pd.Timestamp | None:
    """Latest cached timestamp, or None if the dataset has no cache yet."""
    files = sorted(_dataset_dir(root, dataset).glob("*.parquet"))
    if not files:
        return None
    return pd.read_parquet(files[-1]).index.max()
