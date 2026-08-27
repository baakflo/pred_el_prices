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

    pegel = sub.add_parser(
        "archive-pegel",
        help="Archive PEGELONLINE gauge readings (Rhine at Kaub; rolling ~31-day API window)",
    )
    pegel.add_argument(
        "--archive-dir", type=Path, default=Path("data/archive/water"), help="Output directory"
    )

    efc = sub.add_parser(
        "archive-energyforecast",
        help="Archive today's pre-auction energyforecast.de benchmark forecast (DE-LU)",
    )
    efc.add_argument(
        "--archive-dir", type=Path, default=Path("data/archive/benchmarks"), help="Output directory"
    )
    efc.add_argument(
        "--late",
        action="store_true",
        help="Write a separate _late snapshot (last pre-gate vintage); refused past the gate",
    )

    esnap = sub.add_parser(
        "archive-entsoe-forecasts",
        help="Snapshot tomorrow's ENTSO-E day-ahead load + wind/solar forecasts as published",
    )
    esnap.add_argument(
        "--archive-dir", type=Path, default=Path("data/archive/forecasts"), help="Output directory"
    )

    ecmwf = sub.add_parser(
        "backfill-ecmwf",
        help="Backfill ECMWF open-data ENS runs from the AWS archive (available from 2023-01-18)",
    )
    ecmwf.add_argument("--start", type=date.fromisoformat, required=True, help="First run date")
    ecmwf.add_argument(
        "--end",
        type=date.fromisoformat,
        default=datetime.now(UTC).date(),
        help="Last run date (default: today)",
    )
    ecmwf.add_argument(
        "--archive-dir", type=Path, default=Path("data/archive/weather"), help="Output directory"
    )
    ecmwf.add_argument(
        "--run-hour",
        type=int,
        choices=[0, 12],
        default=0,
        help="Synoptic run hour: 0 (primary) or 12 (evening-before fallback vintage)",
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
    fetch.add_argument("--cache-dir", type=Path, default=Path("data/cache"), help="Cache root")

    capacity = sub.add_parser(
        "fetch-capacity",
        help="Update the monthly installed wind/solar capacity cache (energy-charts.info)",
    )
    capacity.add_argument("--cache-dir", type=Path, default=Path("data/cache"), help="Cache root")

    fuels = sub.add_parser(
        "fetch-fuels", help="Update the daily fuel/carbon price cache (Yahoo proxies)"
    )
    fuels.add_argument("--start", default="2015-01-01", help="UTC start date")
    fuels.add_argument("--cache-dir", type=Path, default=Path("data/cache"), help="Cache root")

    smard = sub.add_parser(
        "fetch-smard", help="Update the SMARD caches (keyless: prices, load, wind/solar)"
    )
    smard.add_argument("--datasets", nargs="+", default=None, help="Subset (default: all)")
    smard.add_argument("--start", default="2015-01-01", help="UTC start date")
    smard.add_argument("--cache-dir", type=Path, default=Path("data/cache"), help="Cache root")

    report = sub.add_parser("report-qa", help="Build the data-QA report page from the cache")
    report.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    report.add_argument("--out", type=Path, default=Path("reports/data_qa"))

    build = sub.add_parser(
        "build-dataset", help="Build the leakage-safe hourly feature/target table"
    )
    build.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    build.add_argument("--out", type=Path, default=Path("data/dataset/hourly.parquet"))

    fc = sub.add_parser(
        "forecast",
        help="Produce the daily pre-gate forecast for the next UTC day (site JSON + log)",
    )
    fc.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    fc.add_argument("--archive-dir", type=Path, default=Path("data/archive/weather"))
    fc.add_argument("--features", type=Path, default=Path("data/dataset/ens_features.parquet"))
    fc.add_argument("--out", type=Path, default=Path("data/site"))
    fc.add_argument(
        "--delivery-day",
        type=date.fromisoformat,
        default=None,
        help="Override the delivery day (default: tomorrow UTC); for testing",
    )
    fc.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Use existing caches/archive without network refresh; for testing",
    )
    fc.add_argument(
        "--allow-ens-fallback",
        action="store_true",
        help="If the 00Z ENS run is unavailable, use the pre-archived 12Z run "
        "of the previous day (staler weather; for late retry slots)",
    )
    fc.add_argument(
        "--refresh-only",
        action="store_true",
        help="Refresh prices and rewrite the site JSON (fill actuals, score "
        "completed days) without ever generating a forecast; for post-auction slots",
    )

    runx = sub.add_parser("run", help="Run a named experiment (artifacts land in runs/)")
    runx.add_argument("name", help="Experiment name, e.g. lear-de")
    runx.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="Override experiment parameters (repeatable)",
    )

    args = parser.parse_args()
    if args.command == "archive-weather":
        from pred_el_prices.pipeline.dwd import archive_run

        archive_run(args.date, args.archive_dir)
    elif args.command == "archive-pegel":
        from pred_el_prices.pipeline.pegel import archive_window

        written = archive_window(args.archive_dir)
        print(f"pegel-kaub: {len(written)} day file(s) written")
    elif args.command == "archive-energyforecast":
        from pred_el_prices.config import energyforecast_token
        from pred_el_prices.pipeline.energyforecast import archive_snapshot

        written = archive_snapshot(args.archive_dir, energyforecast_token(), late=args.late)
        print(f"energyforecast: {written if written else 'already archived today'}")
    elif args.command == "archive-entsoe-forecasts":
        from entsoe import EntsoePandasClient

        from pred_el_prices.config import entsoe_api_key
        from pred_el_prices.pipeline.entsoe_snapshot import archive_snapshot

        client = EntsoePandasClient(api_key=entsoe_api_key())
        written = archive_snapshot(args.archive_dir, client)
        print(f"entsoe-forecasts: {written if written else 'nothing written'}")
    elif args.command == "backfill-ecmwf":
        from pred_el_prices.pipeline.ecmwf import backfill

        backfill(args.start, args.end, args.archive_dir, args.run_hour)
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
    elif args.command == "fetch-capacity":
        from pred_el_prices.pipeline.capacity import update_cache

        df = update_cache(args.cache_dir)
        print(f"installed_power: {len(df)} months through {df.index.max():%Y-%m}")
    elif args.command == "fetch-fuels":
        import pandas as pd

        from pred_el_prices.pipeline.fuels import update_cache

        n = update_cache(args.cache_dir, pd.Timestamp(args.start, tz="UTC"))
        print(f"fuels_daily: {n} rows fetched")
    elif args.command == "fetch-smard":
        import pandas as pd

        from pred_el_prices.pipeline.smard import DATASETS as SMARD_DATASETS
        from pred_el_prices.pipeline.smard import update_cache

        datasets = args.datasets or list(SMARD_DATASETS)
        unknown = set(datasets) - set(SMARD_DATASETS)
        if unknown:
            parser.error(f"unknown datasets: {sorted(unknown)}; choose from {list(SMARD_DATASETS)}")
        for dataset in datasets:
            n = update_cache(args.cache_dir, dataset, pd.Timestamp(args.start, tz="UTC"))
            print(f"{dataset}: {n} rows fetched", flush=True)
    elif args.command == "report-qa":
        from pred_el_prices.reporting.build import build_qa_report

        page = build_qa_report(args.cache_dir, args.out)
        print(f"report written: {page}")
    elif args.command == "forecast":
        from pred_el_prices.daily_forecast import run_daily

        run_daily(
            cache_dir=args.cache_dir,
            archive_dir=args.archive_dir,
            features_path=args.features,
            out_dir=args.out,
            delivery_day=args.delivery_day,
            skip_fetch=args.skip_fetch,
            allow_ens_fallback=args.allow_ens_fallback,
            refresh_only=args.refresh_only,
        )
    elif args.command == "run":
        import json

        from pred_el_prices.experiments import run as run_experiment

        params = {}
        for item in args.overrides:
            key, _, value = item.partition("=")
            try:
                params[key.replace("-", "_")] = json.loads(value)
            except json.JSONDecodeError:
                params[key.replace("-", "_")] = value
        run_experiment(args.name, params)
    elif args.command == "build-dataset":
        import json

        from pred_el_prices.features.dataset import write_dataset

        summary = write_dataset(args.cache_dir, args.out)
        print(json.dumps(summary, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
