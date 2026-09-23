"""Offline tests for the ENTSO-E REMIT outage downloader (fake client, no network)."""

import json

import pandas as pd
from entsoe.exceptions import NoMatchingDataError

from pred_el_prices.pipeline import outages


def _raw(rows: list[dict]) -> pd.DataFrame:
    """Build a fake raw entsoe-py unavailability frame: tz-aware UTC
    created_doc_time index, like `query_unavailability_of_generation_units`."""
    df = pd.DataFrame(rows)
    df["created_doc_time"] = pd.to_datetime(df["created_doc_time"]).dt.tz_localize("UTC")
    df["start"] = pd.to_datetime(df["start"]).dt.tz_localize("UTC")
    df["end"] = pd.to_datetime(df["end"]).dt.tz_localize("UTC")
    return df.set_index("created_doc_time")


def _row(
    mrid="m1",
    revision=1,
    docstatus=None,
    plant_type="Fossil Gas",
    start="2020-01-05",
    end="2020-01-06",
    avail_qty="10.0",
    nominal_power=100.0,
    created_doc_time="2020-01-01",
) -> dict:
    return {
        "mrid": mrid,
        "revision": revision,
        "docstatus": docstatus,
        "businesstype": "Unplanned outage",
        "plant_type": plant_type,
        "production_resource_id": "RES1",
        "production_resource_name": "Plant 1",
        "nominal_power": nominal_power,
        "start": start,
        "end": end,
        "resolution": "PT1M",
        "avail_qty": avail_qty,
        "qty_uom": "MAW",
        "created_doc_time": created_doc_time,
    }


class FakeClient:
    """Records every call; serves canned frames keyed by (zone, month) or mRID."""

    def __init__(self, latest=None, withdrawn=None, revisions=None):
        self.latest = latest or {}
        self.withdrawn = withdrawn or {}
        self.revisions = revisions or {}
        self.calls = []  # (zone, "%Y-%m" of start, docstatus, mRID)

    def query_unavailability_of_generation_units(self, zone, start, end, docstatus=None, mRID=None):
        self.calls.append((zone, f"{start:%Y-%m}", docstatus, mRID))
        if mRID is not None:
            df = self.revisions.get(mRID)
            if isinstance(df, Exception):
                raise df
        elif docstatus == "A13":
            df = self.withdrawn.get((zone, f"{start:%Y-%m}"))
        else:
            df = self.latest.get((zone, f"{start:%Y-%m}"))
        if df is None or df.empty:
            raise NoMatchingDataError()
        return df


class TestFetchMonth:
    def test_tidy_output_types(self):
        raw = _raw([_row(mrid="m1", revision=1, avail_qty="10.5", nominal_power=100.0)])
        client = FakeClient(latest={("DE_LU", "2020-01"): raw})
        df = outages.fetch_month(
            client, "DE_LU", pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-02-01", tz="UTC")
        )
        assert list(df.columns) == outages.TIDY_COLUMNS
        assert df["avail_qty"].dtype == float
        assert df["avail_qty"].iloc[0] == 10.5
        assert df["nominal_power"].dtype == float
        assert pd.api.types.is_integer_dtype(df["revision"])
        assert str(df["created_doc_time"].dt.tz) == "UTC"
        assert str(df["start"].dt.tz) == "UTC"
        assert (df["zone"] == "DE_LU").all()

    def test_withdrawn_rows_included_with_docstatus(self):
        active = _raw([_row(mrid="m1", revision=1)])
        withdrawn = _raw([_row(mrid="m2", revision=1, docstatus="Cancelled")])
        client = FakeClient(
            latest={("FR", "2020-01"): active}, withdrawn={("FR", "2020-01"): withdrawn}
        )
        df = outages.fetch_month(
            client, "FR", pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-02-01", tz="UTC")
        )
        assert set(df["mrid"]) == {"m1", "m2"}
        assert df.set_index("mrid").loc["m2", "docstatus"] == "Cancelled"


class TestBackfill:
    def test_mrid_query_only_for_multirevision_nonwind_solar(self, tmp_path):
        latest = _raw(
            [
                _row(mrid="a", revision=1, plant_type="Fossil Gas"),  # single revision: skip
                _row(mrid="b", revision=2, plant_type="Fossil Gas"),  # candidate
                _row(mrid="c", revision=3, plant_type="Wind Onshore"),  # wind: skip
                _row(mrid="d", revision=2, plant_type="Solar"),  # solar: skip
            ]
        )
        revisions_b = _raw([_row(mrid="b", revision=1), _row(mrid="b", revision=2)])
        client = FakeClient(latest={("DE_LU", "2020-01"): latest}, revisions={"b": revisions_b})

        outages.backfill(
            client,
            ["DE_LU"],
            pd.Timestamp("2020-01-01", tz="UTC"),
            pd.Timestamp("2020-02-01", tz="UTC"),
            tmp_path,
            sleep_s=0,
            revisions_since=pd.Timestamp("2020-01-01", tz="UTC"),
        )

        mrid_calls = [c[3] for c in client.calls if c[3] is not None]
        assert mrid_calls == ["b"]
        rev_path = tmp_path / "outages" / "DE_LU" / "revisions" / "2020-01.parquet"
        assert rev_path.exists()
        assert set(pd.read_parquet(rev_path)["mrid"]) == {"b"}

    def test_already_fetched_mrid_skipped_next_month(self, tmp_path):
        doc_b = lambda: _row(mrid="b", revision=2, plant_type="Fossil Gas")
        client = FakeClient(
            latest={("DE_LU", "2020-01"): _raw([doc_b()]), ("DE_LU", "2020-02"): _raw([doc_b()])},
            revisions={"b": _raw([_row(mrid="b", revision=1), _row(mrid="b", revision=2)])},
        )

        outages.backfill(
            client,
            ["DE_LU"],
            pd.Timestamp("2020-01-01", tz="UTC"),
            pd.Timestamp("2020-03-01", tz="UTC"),
            tmp_path,
            sleep_s=0,
            revisions_since=pd.Timestamp("2020-01-01", tz="UTC"),
        )

        mrid_calls = [c[3] for c in client.calls if c[3] is not None]
        assert mrid_calls == ["b"]  # not re-queried in February

    def test_resume_skips_existing_months_except_last(self, tmp_path):
        latest_dir = tmp_path / "outages" / "DE_LU" / "latest"
        latest_dir.mkdir(parents=True)
        empty = pd.DataFrame(columns=outages.TIDY_COLUMNS)
        empty.to_parquet(latest_dir / "2020-01.parquet")
        empty.to_parquet(latest_dir / "2020-02.parquet")

        client = FakeClient(
            latest={
                ("DE_LU", "2020-02"): _raw([_row(mrid="x", revision=1)]),
                ("DE_LU", "2020-03"): _raw([_row(mrid="y", revision=1)]),
            }
        )

        outages.backfill(
            client,
            ["DE_LU"],
            pd.Timestamp("2020-01-01", tz="UTC"),
            pd.Timestamp("2020-04-01", tz="UTC"),
            tmp_path,
            sleep_s=0,
        )

        months_queried = sorted({c[1] for c in client.calls if c[3] is None})
        assert months_queried == ["2020-02", "2020-03"]  # January (fully cached) skipped

    def test_revisions_since_gates_pre_cutoff_months(self, tmp_path):
        # Multi-revision, non-wind/solar doc in a month well before the
        # default revisions_since cutoff: latest-only, no mRID query.
        latest = _raw([_row(mrid="b", revision=2, plant_type="Fossil Gas")])
        client = FakeClient(latest={("DE_LU", "2020-01"): latest})

        outages.backfill(
            client,
            ["DE_LU"],
            pd.Timestamp("2020-01-01", tz="UTC"),
            pd.Timestamp("2020-02-01", tz="UTC"),
            tmp_path,
            sleep_s=0,
        )  # default revisions_since (2025-10-01) not reached

        mrid_calls = [c[3] for c in client.calls if c[3] is not None]
        assert mrid_calls == []
        assert not (tmp_path / "outages" / "DE_LU" / "revisions" / "2020-01.parquet").exists()
        latest_saved = pd.read_parquet(tmp_path / "outages" / "DE_LU" / "latest" / "2020-01.parquet")
        assert set(latest_saved["mrid"]) == {"b"}

    def test_failed_mrid_is_queued_and_retried_next_run(self, tmp_path):
        latest = _raw([_row(mrid="b", revision=2, plant_type="Fossil Gas")])
        client = FakeClient(
            latest={("DE_LU", "2020-01"): latest}, revisions={"b": RuntimeError("boom")}
        )
        kwargs = {
            "start": pd.Timestamp("2020-01-01", tz="UTC"),
            "end": pd.Timestamp("2020-02-01", tz="UTC"),
            "revisions_since": pd.Timestamp("2020-01-01", tz="UTC"),
        }

        outages.backfill(client, ["DE_LU"], kwargs["start"], kwargs["end"], tmp_path, sleep_s=0,
                          revisions_since=kwargs["revisions_since"])

        revisions_dir = tmp_path / "outages" / "DE_LU" / "revisions"
        pending = json.loads((revisions_dir / "_pending.json").read_text())
        assert "b" in pending
        assert not (revisions_dir / "2020-01.parquet").exists()

        # mRID now succeeds: the next run should recover it from the queue,
        # not lose it because January is otherwise already "done".
        client.revisions["b"] = _raw([_row(mrid="b", revision=1), _row(mrid="b", revision=2)])
        outages.backfill(client, ["DE_LU"], kwargs["start"], kwargs["end"], tmp_path, sleep_s=0,
                          revisions_since=kwargs["revisions_since"])

        pending = json.loads((revisions_dir / "_pending.json").read_text())
        assert pending == {}
        assert set(pd.read_parquet(revisions_dir / "2020-01.parquet")["mrid"]) == {"b"}
