from datetime import UTC, datetime

import pandas as pd

from pred_el_prices.pipeline import netztransparenz

HEADER = "Datum;von;Zeitzone von;bis;Zeitzone bis;50Hertz (MW);Amprion (MW);TenneT TSO (MW);TransnetBW (MW)"


def _csv(start: str, periods: int) -> str:
    rows = [HEADER]
    for t in pd.date_range(start, periods=periods, freq="15min", tz="UTC"):
        e = t + pd.Timedelta(minutes=15)
        rows.append(f"{t:%Y-%m-%d};{t:%H:%M};UTC;{e:%H:%M};UTC;1,500;2,000;3,000;0,250")
    return "\n".join(rows)


def test_parse_reads_german_decimals_and_utc_index() -> None:
    df = netztransparenz.parse(_csv("2026-09-24 22:00", 2))
    assert list(df.index) == list(
        pd.date_range("2026-09-24 22:00", periods=2, freq="15min", tz="UTC")
    )
    assert df.iloc[0].tolist() == [1.5, 2.0, 3.0, 0.25]


def test_snapshot_writes_once_and_skips_partial_day(tmp_path, monkeypatch) -> None:
    now = datetime(2026, 9, 24, 9, 45, tzinfo=UTC)  # Berlin Sep 24 -> delivery Sep 25 (CEST)

    def full(token, kind, start, end):
        # API returns more than asked (previous day too); only the delivery day counts
        return netztransparenz.parse(_csv("2026-09-23 22:00", 2 * 96))

    monkeypatch.setattr(netztransparenz, "fetch", full)
    dest = netztransparenz.archive_snapshot(tmp_path, token="t", now=now)
    assert (
        dest
        == tmp_path
        / "netztransparenz-vermarktung/2026/netztransparenz-vermarktung_20260925.parquet"
    )
    df = pd.read_parquet(dest)
    assert len(df) == 96
    assert df.index[0] == pd.Timestamp("2026-09-24 22:00", tz="UTC")
    assert "Solar / TenneT TSO (MW)" in df.columns and "Wind / 50Hertz (MW)" in df.columns
    assert (df["fetched_at"] == pd.Timestamp(now)).all()

    assert netztransparenz.archive_snapshot(tmp_path, token="t", now=now) is None  # idempotent

    # tomorrow only half there -> unpublished, no file
    monkeypatch.setattr(
        netztransparenz,
        "fetch",
        lambda token, kind, start, end: netztransparenz.parse(_csv("2026-09-24 22:00", 48)),
    )
    later = datetime(2026, 9, 25, 9, 45, tzinfo=UTC)
    assert netztransparenz.archive_snapshot(tmp_path, token="t", now=later) is None
    assert not (
        tmp_path / "netztransparenz-vermarktung/2026/netztransparenz-vermarktung_20260926.parquet"
    ).exists()
