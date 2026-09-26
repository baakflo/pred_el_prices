"""Backfill archiver for ECMWF open-data ENS forecasts over Germany.

ECMWF publishes its operational open-data forecasts (51-member ENS, 0.4 deg
until Feb 2024, 0.25 deg after) on a public AWS bucket with history back to
2023-01-18 — unlike DWD's 24 h window, this archive is backfillable. Each
3-hourly step is one global GRIB (~2.5 GB) holding all members and params,
but a .index sidecar gives byte offsets per field, so we range-request only
the Germany-relevant surface fields (~0.8-2.7 GB/date instead of ~25 GB).

Parameter availability grows over time: 10u/10v/2t from the archive start,
ssrd (accumulated J/m^2 since forecast start; de-accumulate downstream) and
100u/100v from mid-March 2024. Missing params are skipped silently, so early
dates yield wind/temp spread only.

The 00Z run is published ~7-8 h after synoptic time, well before the 12:00
CET day-ahead auction on D-1 (the 06Z run is not — do not use it). The 12Z
run (published ~20:00 UTC) serves as the evening-before fallback vintage.
Output mirrors the ICON archiver (dwd.py): per-member means over 1-degree
cells covering Germany, one Parquet per run.
"""

from __future__ import annotations

import email.parser
import json
import os
import random
import tempfile
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from pred_el_prices.pipeline.dwd import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, aggregate_members

# Identical files are published to several channels; a throttled attempt
# moves on to the next mirror instead of re-queueing behind the same one.
# Order is freshness-first, then answers-first: data.ecmwf.int and S3 both
# carry a new run right at publication, but S3 deterministically 503s ever
# more clients (cloud IPs and non-curl UAs since ~2026-08-18; by 2026-09-02
# even residential curl got an instant 503, while data.ecmwf.int served the
# same file in 1.7s). A fast 503 costs nothing, but first position means a
# stalling throttle burns its 180s timeout on EVERY request of the sweep —
# so the mirror that demonstrably answers goes first and S3 becomes the
# backup. GCS syncs hours later (GCS-first broke the same-day 12Z fallback
# job on 2026-08-26 — both evening slots 404'd the fresh run), so it is
# last — but crucial for backfills: it keeps the full archive (probed back
# to 2024-03-19) with no UA filtering or cloud-IP throttling, while
# data.ecmwf.int holds only the last few days (historical dates 404 there;
# the all-mirrors-404 rule below keeps backfill semantics intact). The
# Azure mirror requires a SAS token since 2026-08 (409 without) — not
# listed.
MIRRORS = [
    "https://data.ecmwf.int/forecasts",
    "https://ecmwf-forecasts.s3.amazonaws.com",
    "https://storage.googleapis.com/ecmwf-open-data",
]

# Bucket layout changed twice; earliest layout first tried last.
MODEL_PATHS = ["ifs/0p25", "0p25", "0p4-beta"]

# GRIB shortName -> our variable name (matching dwd.py where the quantity
# matches; ssrd has no ICON twin — ICON splits direct/diffuse).
VARIABLES = {
    "10u": "u_10m",
    "10v": "v_10m",
    "2t": "t_2m",
    "ssrd": "ssrd",
    "100u": "u_100m",
    "100v": "v_100m",
}

# ENS steps are 3-hourly. +21h..+48h from the 00Z run of D-1 covers the
# delivery day D fully in UTC and in CET/CEST; +33h..+60h from the 12Z run
# of D-2 covers the same window (production fallback when the morning 00Z
# download fails — the 12Z is on S3 the previous evening, ~20:00 UTC).
STEPS = {0: range(21, 49, 3), 12: range(33, 61, 3)}


# The buckets throttle bursts of range requests with 503 Slow Down; keep one
# session, pace the requests, and rotate mirrors when throttled. The herd
# pulling a freshly published run can keep a bucket throttled for many
# minutes (observed 2026-08-15..17: three runs exhausted an ~6-minute retry
# budget), so a request rides it out for up to ~15 minutes across mirrors,
# with jittered exponential backoff between mirror sweeps. Time-boxed
# callers (the pre-gate 09:50 slot, which must give up quickly and use the
# 12Z fallback instead) shrink the budget via env var.
_session = requests.Session()
# The buckets deterministically 503 the default python-requests User-Agent
# (observed 2026-08-26 from both residential and cloud IPs; curl/wget-class
# UAs pass). Identify as curl — same anonymous GETs, same pacing/backoff.
_session.headers["User-Agent"] = "curl/8.5.0"
REQUEST_PACING_S = float(os.environ.get("PEP_ENS_PACING_S", "0.5"))
RETRY_BUDGET_S = int(os.environ.get("PEP_ENS_RETRY_BUDGET_S", "900"))


def _get(path: str, headers: dict | None = None) -> requests.Response:
    deadline = time.monotonic() + RETRY_BUDGET_S
    backoff = 5.0
    last: requests.Response | requests.RequestException | None = None
    while True:
        codes = []
        for base in MIRRORS:
            time.sleep(REQUEST_PACING_S)
            try:
                resp = _session.get(f"{base}/{path}", headers=headers, timeout=180)
            except requests.RequestException as e:
                last = e
                continue
            if resp.ok:
                return resp
            codes.append(resp.status_code)
            last = resp
        if codes and all(c == 404 for c in codes):
            last.raise_for_status()  # absent on every mirror: no point retrying
        if time.monotonic() + backoff >= deadline:
            break
        time.sleep(backoff + random.uniform(0, 20))
        backoff = min(backoff * 2, 180)
    if isinstance(last, requests.Response):
        last.raise_for_status()
    raise last


def _step_path(run_date: date, model_path: str, step: int, suffix: str, run_hour: int = 0) -> str:
    stamp = f"{run_date:%Y%m%d}"
    return (
        f"{stamp}/{run_hour:02d}z/{model_path}/enfo/"
        f"{stamp}{run_hour:02d}0000-{step}h-enfo-ef.{suffix}"
    )


def _discover_model_path(run_date: date, run_hour: int = 0) -> str:
    first_step = STEPS[run_hour][0]
    for candidate in MODEL_PATHS:
        try:
            _get(_step_path(run_date, candidate, first_step, "index", run_hour))
            return candidate
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                continue
            raise
    raise FileNotFoundError(f"no {run_hour:02d}Z ENS index found for {run_date} in {MODEL_PATHS}")


def _wanted_ranges(index_text: str) -> list[list[int]]:
    """Byte ranges of the wanted surface fields, adjacent ranges merged."""
    offsets = []
    for line in index_text.splitlines():
        if not line.strip():
            continue
        field = json.loads(line)
        if field.get("param") in VARIABLES and field.get("levtype") == "sfc":
            offsets.append((field["_offset"], field["_offset"] + field["_length"]))
    ranges: list[list[int]] = []
    for start, end in sorted(offsets):
        if ranges and start == ranges[-1][1]:
            ranges[-1][1] = end
        else:
            ranges.append([start, end])
    return ranges


# The ~280 wanted fields per step are scattered across a 6.6 GB file, so one
# range per request meant ~280 paced requests per step (~200 s; ~40 min per
# run). data.ecmwf.int answers multi-range requests (206 multipart/byteranges;
# 2026-09-25: 10 requests, 65 s, byte-identical), cutting the request count
# ~28x at the same pacing. S3 ignores multi-range (200 with the whole file)
# and GCS rejects it (400), so a batch goes to data.ecmwf.int only, and any
# failure falls back to the per-range path with its mirror/backoff logic.
RANGES_PER_REQUEST = 30


def _get_multi(path: str, ranges: list[list[int]]) -> list[bytes] | None:
    """One multi-range request to the first mirror; None on anything unexpected."""
    time.sleep(REQUEST_PACING_S)
    spec = ",".join(f"{start}-{end - 1}" for start, end in ranges)
    try:
        resp = _session.get(
            f"{MIRRORS[0]}/{path}", headers={"Range": f"bytes={spec}"}, timeout=180, stream=True
        )
    except requests.RequestException:
        return None
    ctype = resp.headers.get("Content-Type", "")
    if resp.status_code != 206 or not ctype.startswith("multipart/byteranges"):
        resp.close()  # never read a whole-file 200 body
        return None
    try:
        msg = email.parser.BytesParser().parsebytes(
            b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + resp.content
        )
        parts = {}
        for part in msg.get_payload():
            first = part["Content-Range"].split()[1].split("-")[0]
            parts[int(first)] = part.get_payload(decode=True)
        chunks = [parts[start] for start, _ in ranges]
    except (requests.RequestException, KeyError, ValueError, IndexError, AttributeError):
        return None
    if [len(c) for c in chunks] != [end - start for start, end in ranges]:
        return None
    return chunks


def _fetch_step(run_date: date, model_path: str, step: int, run_hour: int = 0) -> bytes:
    index = _get(_step_path(run_date, model_path, step, "index", run_hour)).text
    grib_path = _step_path(run_date, model_path, step, "grib2", run_hour)
    ranges = _wanted_ranges(index)
    chunks: list[bytes] = []
    for i in range(0, len(ranges), RANGES_PER_REQUEST):
        batch = ranges[i : i + RANGES_PER_REQUEST]
        got = _get_multi(grib_path, batch) if len(batch) > 1 else None
        if got is None:
            got = [
                _get(grib_path, headers={"Range": f"bytes={start}-{end - 1}"}).content
                for start, end in batch
            ]
        chunks.extend(got)
    return b"".join(chunks)


def _open_var(path: Path, short_name: str) -> xr.DataArray | None:
    """One variable across all members; control (if present) becomes member 0.

    Mixed-height surface fields (2 m temp vs 10 m wind) cannot share a cfgrib
    hypercube, so each variable is opened with its own shortName filter.
    """

    def _open(data_type: str) -> xr.DataArray | None:
        ds = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={
                "indexpath": "",
                "filter_by_keys": {"dataType": data_type, "shortName": short_name},
            },
        )
        if not ds.data_vars:
            return None
        (name,) = ds.data_vars
        return ds[name]

    pf = _open("pf")
    if pf is None:
        return None
    cf = _open("cf")
    if cf is None:  # newer files carry no control fields
        return pf
    return xr.concat([cf.expand_dims(number=[0]), pf], dim="number", coords="minimal")


def _aggregate_step(raw_grib: bytes, tmp_dir: Path) -> pd.DataFrame:
    path = tmp_dir / "step.grib2"
    path.write_bytes(raw_grib)
    frames = []
    for short_name, our_name in VARIABLES.items():
        da = _open_var(path, short_name)
        if da is None:
            continue
        de = da.sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))
        lat2d, lon2d = np.meshgrid(de.latitude.values, de.longitude.values, indexing="ij")
        fields = de.values.reshape(de.sizes["number"], -1)
        df = aggregate_members(lat2d.ravel(), lon2d.ravel(), fields)
        df["variable"] = our_name
        frames.append(df)
        de.close()
    return pd.concat(frames, ignore_index=True)


def archive_run(run_date: date, archive_dir: Path, run_hour: int = 0) -> Path:
    """Download and aggregate one ENS run from the AWS archive. Idempotent."""
    out = (
        archive_dir
        / "ecmwf-ens"
        / f"{run_date:%Y}"
        / f"ecmwf-ens_{run_date:%Y%m%d}{run_hour:02d}.parquet"
    )
    if out.exists():
        print(f"already archived: {out}")
        return out

    model_path = _discover_model_path(run_date, run_hour)
    run_time = datetime(run_date.year, run_date.month, run_date.day, run_hour, tzinfo=UTC)
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for step in STEPS[run_hour]:
            df = _aggregate_step(_fetch_step(run_date, model_path, step, run_hour), tmp_dir)
            df["valid_time"] = run_time + timedelta(hours=step)
            frames.append(df)
            print(f"step +{step}h done ({df['variable'].nunique()} vars)", flush=True)

    result = pd.concat(frames, ignore_index=True)
    result["run_time"] = run_time
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    print(f"archived {len(result)} rows -> {out}")
    return out


def backfill(start: date, end: date, archive_dir: Path, run_hour: int = 0) -> None:
    """Archive every run date in [start, end]; log and continue on missing dates."""
    day = start
    while day <= end:
        try:
            archive_run(day, archive_dir, run_hour)
        except FileNotFoundError as e:
            print(f"SKIP {day}: {e}", flush=True)
        day += timedelta(days=1)
