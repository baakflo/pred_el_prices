"""Command-line entry point. Target (Phase 0 deliverable): one command produces a clean
feature/target table from raw APIs, e.g. `pep build-dataset`."""

import argparse
from datetime import UTC, date, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="pep", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    archive = sub.add_parser(
        "archive-weather",
        help="Archive today's 00Z ICON-EU-EPS ensemble run (per-member regional aggregates)",
    )
    archive.add_argument(
        "--date",
        type=date.fromisoformat,
        default=datetime.now(UTC).date(),
        help="Run date in UTC (default: today)",
    )
    archive.add_argument(
        "--archive-dir", type=Path, default=Path("data/archive/weather"), help="Output directory"
    )

    fetch = sub.add_parser(
        "fetch-entsoe",
        help="Backfill/update the local ENTSO-E Parquet cache (resumes where it left off)",
    )
    fetch.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Subset of datasets (default: all)",
    )
    fetch.add_argument("--start", default="2015-01-01", help="UTC start date")
    fetch.add_argument("--end", default=None, help="UTC end date (default: now)")
    fetch.add_argument(
        "--cache-dir", type=Path, default=Path("data/cache/entsoe"), help="Cache root"
    )

    report = sub.add_parser("report-qa", help="Build the data-QA report page from the cache")
    report.add_argument("--cache-dir", type=Path, default=Path("data/cache/entsoe"))
    report.add_argument("--out", type=Path, default=Path("reports/data_qa"))

    args = parser.parse_args()
    if args.command == "archive-weather":
        from pred_el_prices.pipeline.dwd import archive_run

        archive_run(args.date, args.archive_dir)
    elif args.command == "fetch-entsoe":
        import pandas as pd
        from entsoe import EntsoePandasClient

        from pred_el_prices.config import entsoe_api_key
        from pred_el_prices.pipeline.entsoe import DATASETS, backfill

        datasets = args.datasets or list(DATASETS)
        unknown = set(datasets) - set(DATASETS)
        if unknown:
            parser.error(f"unknown datasets: {sorted(unknown)}; choose from {list(DATASETS)}")
        start = pd.Timestamp(args.start, tz="UTC")
        end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC")
        client = EntsoePandasClient(api_key=entsoe_api_key())
        backfill(client, datasets, start, end, args.cache_dir)
    elif args.command == "report-qa":
        from pred_el_prices.reporting.build import build_qa_report

        page = build_qa_report(args.cache_dir, args.out)
        print(f"report written: {page}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
