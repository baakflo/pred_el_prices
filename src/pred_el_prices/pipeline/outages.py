"""ENTSO-E REMIT generation-unit outage downloader (documentType A80).

A plain query only ever returns the *latest* revision of each document;
passing a specific `mRID` returns every revision of that one document. A
plain query also omits withdrawn documents -- those need a separate
docstatus="A13" query. Cached per zone at native (per-document,
per-availability-point) grain:

    <cache_root>/outages/<zone>/latest/<YYYY-MM>.parquet     -- one month's
        latest known state of every document touching that month
    <cache_root>/outages/<zone>/revisions/<YYYY-MM>.parquet  -- full revision
        history of the (non-wind/solar) multi-revision documents discovered
        in that month
    <cache_root>/outages/<zone>/installed_capacity.parquet   -- yearly
        installed capacity per production type, for normalizing outage MW

Revision history is expensive (one request per document) and only pulled for
documents that actually changed (latest revision > 1) and are not wind/solar
(REMIT outage disclosure for those is sparse and not the scarcity signal we
need); see docs/pred_el_prices_project_plan.md.
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
)

from pred_el_prices.pipeline.entsoe import _is_retryable, month_ranges

PENDING_FILE = "_pending.json"

TIDY_COLUMNS = [
    "zone",
    "mrid",
    "revision",
    "created_doc_time",
    "docstatus",
    "businesstype",
    "plant_type",
    "production_resource_id",
    "production_resource_name",
    "nominal_power",
    "start",
    "end",
    "resolution",
    "avail_qty",
    "qty_uom",
]

# PSR type names entsoe-py maps B16/B18/B19 to (see entsoe.mappings.PSRTYPE_MAPPINGS).
WIND_SOLAR_PLANT_TYPES = {"Solar", "Wind Offshore", "Wind Onshore"}

# Per-mRID revision fetching is one request per changed document, so it does
# not scale to the full 2018- history: default cutoff is the platform
# migration that made created_doc_time meaningful for revision timing (see
# module docstring in the outage_probe prototype). Earlier months store
# latest-version rows only.
DEFAULT_REVISIONS_SINCE = pd.Timestamp("2025-10-01", tz="UTC")


def _wait_policy(retry_state):
    """Exponential backoff, but a 429 honours Retry-After (or backs off
    harder than a transient 5xx) -- an unattended multi-hour backfill can
    afford to wait out sustained throttling rather than give up on a month.
    """
    default = wait_exponential(multiplier=5, max=300)(retry_state)
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code == 429:
        retry_after = exc.response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(default, float(retry_after))
            except ValueError:
                pass
        return max(default, 30.0 * retry_state.attempt_number)
    return default


def _describe_error(exc: BaseException) -> str:
    """Short, secret-free description for warning logs (never the request
    URL, which carries the API token as a query parameter)."""
    response = getattr(exc, "response", None)
    if response is not None:
        return f"HTTP {response.status_code}"
    return type(exc).__name__


@retry(
    stop=stop_after_attempt(8) | stop_after_delay(900),
    wait=_wait_policy,
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
def _call_outages(client, zone, start, end, docstatus=None, mRID=None):
    return client.query_unavailability_of_generation_units(
        zone, start=start, end=end, docstatus=docstatus, mRID=mRID
    )


@retry(
    stop=stop_after_attempt(8) | stop_after_delay(900),
    wait=_wait_policy,
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
def _call_capacity(client, zone, start, end):
    return client.query_installed_generation_capacity(zone, start=start, end=end)


def _tidy(df: pd.DataFrame, zone: str) -> pd.DataFrame:
    """Raw entsoe-py unavailability frame -> flat UTC tidy frame, one row per
    (document version x availability-quantity segment)."""
    df = df.reset_index()  # created_doc_time back from the index to a column
    df["created_doc_time"] = df["created_doc_time"].dt.tz_convert("UTC")
    df["start"] = df["start"].dt.tz_convert("UTC")
    df["end"] = df["end"].dt.tz_convert("UTC")
    df["avail_qty"] = pd.to_numeric(df["avail_qty"], errors="coerce")
    df["nominal_power"] = pd.to_numeric(df["nominal_power"], errors="coerce")
    df["revision"] = df["revision"].astype(int)
    df["zone"] = zone
    return df[TIDY_COLUMNS].sort_values(["mrid", "revision", "start"]).reset_index(drop=True)


def fetch_month(
    client: EntsoePandasClient, zone: str, month_start: pd.Timestamp, month_end: pd.Timestamp
) -> pd.DataFrame:
    """One month of A80 outage documents: latest revision of every active
    document, plus withdrawn (docstatus A13) documents a plain query omits.

    Empty tidy frame (right columns, zero rows) if there is no data at all.
    """
    parts = []
    for docstatus in (None, "A13"):
        try:
            parts.append(_call_outages(client, zone, month_start, month_end, docstatus=docstatus))
        except NoMatchingDataError:
            continue
    if not parts:
        return pd.DataFrame(columns=TIDY_COLUMNS)
    return _tidy(pd.concat(parts), zone)


def _candidate_mrids(latest: pd.DataFrame) -> list[str]:
    """Documents worth a full revision-history fetch: latest revision > 1
    (i.e. it actually changed after first publication) and not wind/solar."""
    if latest.empty:
        return []
    by_doc = latest.groupby("mrid").agg(revision=("revision", "max"), plant_type=("plant_type", "first"))
    mask = (by_doc["revision"] > 1) & (~by_doc["plant_type"].isin(WIND_SOLAR_PLANT_TYPES))
    return by_doc.index[mask].tolist()


def _fetch_revisions(
    client: EntsoePandasClient,
    zone: str,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    mrid: str,
) -> pd.DataFrame:
    """All revisions of one document.

    The plain month window is NOT enough: a document's *earlier* revisions
    can declare a different (usually shorter/earlier) availability period
    than the latest one, so they fall outside [month_start, month_end) and
    the API silently omits them -- verified empirically (a doc with 8
    revisions returned only 3..8 for the plain month window). A window
    reaching back to month_start - 1y + 1month (span exactly one year, so
    entsoe-py's >1y auto year-splitting -- and its extra HTTP requests --
    never kicks in) reliably recovers revision 1 too. If the platform still
    rejects the span (HTTP 400), fall back to the plain month window rather
    than losing the document entirely (its earliest revisions may be
    missing in that case).
    """
    wide_start = month_start - pd.DateOffset(years=1) + pd.DateOffset(months=1)
    try:
        raw = _call_outages(client, zone, wide_start, month_end, mRID=mrid)
    except NoMatchingDataError:
        return pd.DataFrame(columns=TIDY_COLUMNS)
    except requests.HTTPError as exc:
        if exc.response is None or exc.response.status_code != 400:
            raise
        try:
            raw = _call_outages(client, zone, month_start, month_end, mRID=mrid)
        except NoMatchingDataError:
            return pd.DataFrame(columns=TIDY_COLUMNS)
    return _tidy(raw, zone)


def _write_revisions(revisions_dir: Path, label: str, df: pd.DataFrame) -> None:
    """Merge new revision rows into that month's file (idempotent)."""
    if df.empty:
        return
    path = revisions_dir / f"{label}.parquet"
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
        df = df.drop_duplicates(["mrid", "revision", "start"])
    df.to_parquet(path)


def _load_pending(revisions_dir: Path) -> dict:
    """mRIDs whose revision fetch failed on an earlier run, with the month
    window to retry them against (so a failure is retried, not lost)."""
    path = revisions_dir / PENDING_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_pending(revisions_dir: Path, pending: dict) -> None:
    (revisions_dir / PENDING_FILE).write_text(json.dumps(pending))


def _existing_mrids(revisions_dir: Path) -> set[str]:
    """mRIDs already on disk for this zone, across every revisions file.

    Tolerant of a concurrent writer (a sibling process backfilling a
    different month range for the same zone): a file that fails to read
    (e.g. caught mid-write) is skipped rather than crashing the run.
    """
    mrids: set[str] = set()
    for path in sorted(revisions_dir.glob("*.parquet")):
        try:
            mrids.update(pd.read_parquet(path, columns=["mrid"])["mrid"].unique())
        except Exception:  # noqa: BLE001, S112 - best-effort read of a possibly-racing file
            continue
    return mrids


def backfill(
    client: EntsoePandasClient,
    zones: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_root,
    sleep_s: float = 0.2,
    revisions_since: pd.Timestamp = DEFAULT_REVISIONS_SINCE,
) -> None:
    """Fetch month by month, per zone; resumes from the last cached month.

    For every month, stores the latest known state of every touching
    document. For months >= `revisions_since` only, also pulls full revision
    history for documents that changed (latest revision > 1) and are not
    wind/solar, skipping mRIDs already on disk for that zone (a document can
    span more than one month, and concurrent processes covering other month
    ranges of the same zone may add to that pool mid-run). A failed mRID
    fetch is queued to `revisions/_pending.json` and retried at the start of
    the next call rather than silently lost.
    """
    cache_root = Path(cache_root)
    for zone in zones:
        zone_dir = cache_root / "outages" / zone
        latest_dir = zone_dir / "latest"
        revisions_dir = zone_dir / "revisions"
        latest_dir.mkdir(parents=True, exist_ok=True)
        revisions_dir.mkdir(parents=True, exist_ok=True)

        existing_months = sorted(p.stem for p in latest_dir.glob("*.parquet"))
        resume_month = existing_months[-1] if existing_months else None
        known_mrids = _existing_mrids(revisions_dir)

        pending = _load_pending(revisions_dir)
        if pending:
            still_pending = {}
            n_recovered = 0
            for mrid, window in pending.items():
                w_start, w_end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
                try:
                    revisions = _fetch_revisions(client, zone, w_start, w_end, mrid)
                except Exception as exc:  # noqa: BLE001 - one bad mRID must not lose the queue
                    print(f"  WARNING: {zone} retry mRID {mrid}: {_describe_error(exc)}", flush=True)
                    still_pending[mrid] = window
                    continue
                _write_revisions(revisions_dir, f"{w_start:%Y-%m}", revisions)
                known_mrids.add(mrid)
                n_recovered += 1
                time.sleep(sleep_s)
            pending = still_pending
            _save_pending(revisions_dir, pending)
            print(f"{zone} outages retry-queue: {n_recovered} recovered, {len(pending)} still pending", flush=True)

        for m_start, m_end in month_ranges(start, end):
            label = f"{m_start:%Y-%m}"
            if label in existing_months and label != resume_month:
                continue

            try:
                latest = fetch_month(client, zone, m_start, m_end)
            except Exception as exc:  # noqa: BLE001 - stop this zone cleanly; resumable next run
                print(f"  WARNING: {zone} {label} month fetch failed: {_describe_error(exc)}; "
                      f"stopping {zone} here (resumable)", flush=True)
                break
            latest.to_parquet(latest_dir / f"{label}.parquet")

            candidates = []
            n_ok = 0
            if m_start >= revisions_since:
                candidates = [m for m in _candidate_mrids(latest) if m not in known_mrids]
                for mrid in candidates:
                    try:
                        revisions = _fetch_revisions(client, zone, m_start, m_end, mrid)
                    except Exception as exc:  # noqa: BLE001 - one bad mRID must not lose the month
                        print(f"  WARNING: {zone} {label} mRID {mrid}: {_describe_error(exc)}", flush=True)
                        pending[mrid] = {"start": m_start.isoformat(), "end": m_end.isoformat()}
                        continue
                    _write_revisions(revisions_dir, label, revisions)
                    known_mrids.add(mrid)
                    n_ok += 1
                    time.sleep(sleep_s)
                _save_pending(revisions_dir, pending)

            print(
                f"{zone} outages {label}: {len(latest)} docs, "
                f"{len(candidates)} mRID queries ({n_ok} ok)",
                flush=True,
            )
            time.sleep(sleep_s)


def fetch_installed_capacity(
    client: EntsoePandasClient, zones: list[str], years, cache_root
) -> None:
    """Yearly installed generation capacity per production type, per zone (MW)."""
    cache_root = Path(cache_root)
    for zone in zones:
        rows = []
        for year in years:
            y_start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
            y_end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
            try:
                rows.append(_call_capacity(client, zone, y_start, y_end))
            except NoMatchingDataError:
                continue
        if not rows:
            print(f"{zone} installed_capacity: no data", flush=True)
            continue
        combined = pd.concat(rows).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.index = combined.index.tz_convert("UTC")
        combined.index.name = "year"
        out_dir = cache_root / "outages" / zone
        out_dir.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out_dir / "installed_capacity.parquet")
        print(f"{zone} installed_capacity: {len(combined)} year(s)", flush=True)
