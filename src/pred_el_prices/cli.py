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

    archive_det = sub.add_parser(
        "archive-weather-det",
        help="Archive today's ICON-D2 + ICON-EU deterministic runs (00Z/03Z, regional "
        "aggregates + hub-height wind)",
    )
    archive_det.add_argument(
        "--date",
        type=date.fromisoformat,
        default=datetime.now(UTC).date(),
        help="Run date in UTC (default: today)",
    )
    archive_det.add_argument(
        "--archive-dir", type=Path, default=Path("data/archive/weather"), help="Output directory"
    )
    archive_det.add_argument(
        "--run",
        choices=["00", "03"],
        default=None,
        help="Archive only this run hour (default: both 00Z and 03Z)",
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

    ntsnap = sub.add_parser(
        "archive-netztransparenz",
        help="Snapshot tomorrow's TSO EEG marketing forecast (solar/wind) from netztransparenz.de",
    )
    ntsnap.add_argument(
        "--archive-dir", type=Path, default=Path("data/archive/benchmarks"), help="Output directory"
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
    fetch.add_argument(
        "--zones",
        nargs="+",
        default=None,
        help="Neighbour zones as entsoe-py area codes, e.g. FR NL (default: Germany)",
    )

    outages = sub.add_parser(
        "fetch-outages",
        help="Backfill ENTSO-E REMIT generation-unit outage documents (resumes where it left off)",
    )
    outages.add_argument("--zones", nargs="+", default=["DE_LU", "FR"], help="entsoe-py area codes")
    outages.add_argument("--start", default="2018-12-01", help="UTC start date")
    outages.add_argument("--end", default=None, help="UTC end date (default: now)")
    outages.add_argument("--cache-dir", type=Path, default=Path("data/cache"), help="Cache root")
    outages.add_argument(
        "--revisions-since",
        default="2025-10-01",
        help="Only pull per-document revision history for months on/after this UTC date "
        "(one request per changed document; earlier months store latest-version rows only)",
    )

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

    echarts = sub.add_parser(
        "fetch-energy-charts",
        help="Update the energy-charts.info day-ahead price cache (keyless third outlet)",
    )
    echarts.add_argument("--start", default=None, help="UTC start date (default: 14 days back)")
    echarts.add_argument("--end", default=None, help="UTC end date (default: now)")
    echarts.add_argument("--cache-dir", type=Path, default=Path("data/cache"), help="Cache root")

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
    fc.add_argument(
        "--evening",
        action="store_true",
        help="Evening edition: target the day AFTER tomorrow using the 12Z ENS "
        "run + load surrogate; replaced by the next morning's regular run",
    )
    fc.add_argument(
        "--allow-load-surrogate",
        action="store_true",
        help="If ENTSO-E has no load forecast for the delivery day, publish with "
        "the surrogate model instead of failing (flagged; for retry slots)",
    )

    fc.add_argument(
        "--nets-bundle",
        type=Path,
        default=None,
        help="Network ensemble bundle (pep train-nets): also publish the nets forecast "
        "(additive; LEAR publishes even if the nets step fails)",
    )

    tn = sub.add_parser(
        "train-nets",
        help="Train the weekly network ensemble (12 JSU + 12 quantile) into one bundle file",
    )
    tn.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    tn.add_argument("--out", type=Path, required=True, help="Bundle file to write")
    tn.add_argument(
        "--monday",
        type=date.fromisoformat,
        default=None,
        help="Week the bundle serves; trains on days <= monday - 2 "
        "(default: today if Monday, else the next Monday, UTC)",
    )
    tn.add_argument("--n-jobs", type=int, default=-1)

    sp = sub.add_parser(
        "seed-nets-pit",
        help="Seed the nets' PIT/median history from backtest quantiles (+ nets logs)",
    )
    sp.add_argument("--quantiles", type=Path, required=True, help="Raw quantiles.parquet")
    sp.add_argument("--cal", type=Path, default=None, help="Recalibrated quantiles.parquet (q50)")
    sp.add_argument("--before", required=True, help="Use backtest hours before this UTC day")
    sp.add_argument(
        "--nets-logs", type=Path, nargs="*", default=[], help="nets_log.parquet files to add"
    )
    sp.add_argument("--out", type=Path, required=True, help="Seed parquet to write")

    bn = sub.add_parser(
        "backfill-nets",
        help="Replay the production network path over past days (weekly bundles, own RES)",
    )
    bn.add_argument("--start", required=True, help="First delivery day (UTC)")
    bn.add_argument("--end", required=True, help="Last delivery day (UTC)")
    bn.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    bn.add_argument(
        "--features", type=Path, required=True, help="Production ENS feature table (own RES)"
    )
    bn.add_argument("--pit-seed", type=Path, required=True, help="Seed from seed-nets-pit")
    bn.add_argument("--out", type=Path, required=True)
    bn.add_argument("--lear-log", type=Path, default=None, help="LEAR forecast_log.parquet")
    bn.add_argument("--lear-history", type=Path, default=None, help="Published history.json")
    bn.add_argument("--n-jobs", type=int, default=-1)

    bs = sub.add_parser(
        "build-site",
        help="Rebuild site JSON (latest, history, days/) from existing nets + LEAR logs",
    )
    bs.add_argument("--nets-log", type=Path, required=True, help="nets_log.parquet or nets_log/")
    bs.add_argument("--lear-log", type=Path, required=True, help="LEAR forecast_log.parquet")
    bs.add_argument("--history", type=Path, default=None, help="Published history.json base")
    bs.add_argument("--end", default=None, help="Last delivery day for latest.json (UTC)")
    bs.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    bs.add_argument("--out", type=Path, required=True)

    bf = sub.add_parser(
        "backfill-history",
        help="Fill curve-less site history days from backtest runs (flagged post_gate)",
    )
    bf.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        required=True,
        help="Backtest run directories; the first that covers a day wins",
    )
    bf.add_argument("--out", type=Path, default=Path("data/site"), help="Site JSON directory")
    bf.add_argument("--start", required=True, help="First delivery day (UTC, YYYY-MM-DD)")
    bf.add_argument("--end", required=True, help="Last delivery day (UTC, YYYY-MM-DD)")

    sr = sub.add_parser("score-run", help="Score a forecast run and write scorecard.json into it")
    sr.add_argument("run_dir", type=Path, help="Run directory (contains forecast.parquet)")
    sr.add_argument(
        "--vs", type=Path, default=None, dest="vs_run_dir", help="Baseline run to compare against"
    )
    sr.add_argument("--start", default=None, help="Slice start day, UTC (YYYY-MM-DD, inclusive)")
    sr.add_argument("--end", default=None, help="Slice end day, UTC (YYYY-MM-DD, inclusive)")
    sr.add_argument(
        "--dataset", type=Path, default=Path("data/dataset/hourly.parquet"), help="Price dataset"
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
    elif args.command == "archive-weather-det":
        from pred_el_prices.pipeline.dwd_det import archive_today

        runs = (int(args.run),) if args.run else (0, 3)
        written = archive_today(args.archive_dir, args.date, runs=runs)
        print(f"archive-weather-det: {len(written)} file(s) written")
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
        import requests
        from entsoe import EntsoePandasClient

        from pred_el_prices.config import entsoe_api_key
        from pred_el_prices.pipeline.entsoe_snapshot import archive_snapshot

        client = EntsoePandasClient(api_key=entsoe_api_key())
        # Best-effort: a platform outage must not fail the archive run after
        # the weather already landed — later slots retry the snapshot, and a
        # real gap shows as a missing file, not a lost workflow.
        try:
            written = archive_snapshot(args.archive_dir, client)
        except requests.RequestException as e:
            print(f"::warning::entsoe-forecasts snapshot skipped ({e}); later slots retry")
        else:
            print(f"entsoe-forecasts: {written if written else 'nothing written'}")
    elif args.command == "archive-netztransparenz":
        import requests

        from pred_el_prices.config import netztransparenz_credentials
        from pred_el_prices.pipeline.netztransparenz import access_token, archive_snapshot

        # best-effort like the ENTSO-E snapshot: later slots retry
        try:
            written = archive_snapshot(
                args.archive_dir, access_token(*netztransparenz_credentials())
            )
        except requests.RequestException as e:
            print(f"::warning::netztransparenz snapshot skipped ({e}); later slots retry")
        else:
            print(f"netztransparenz: {written if written else 'nothing written'}")
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
        for zone in args.zones or [None]:
            backfill(client, datasets, start, end, args.cache_dir, zone=zone)
    elif args.command == "fetch-outages":
        import pandas as pd
        from entsoe import EntsoePandasClient

        from pred_el_prices.config import entsoe_api_key
        from pred_el_prices.pipeline.outages import backfill, fetch_installed_capacity

        start = pd.Timestamp(args.start, tz="UTC")
        end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC")
        revisions_since = pd.Timestamp(args.revisions_since, tz="UTC")
        client = EntsoePandasClient(api_key=entsoe_api_key())
        fetch_installed_capacity(client, args.zones, range(2018, 2027), args.cache_dir)
        backfill(client, args.zones, start, end, args.cache_dir, revisions_since=revisions_since)
    elif args.command == "fetch-capacity":
        from pred_el_prices.pipeline.capacity import update_cache

        df = update_cache(args.cache_dir)
        print(f"installed_power: {len(df)} months through {df.index.max():%Y-%m}")
    elif args.command == "fetch-fuels":
        import pandas as pd

        from pred_el_prices.pipeline.fuels import update_cache

        n = update_cache(args.cache_dir, pd.Timestamp(args.start, tz="UTC"))
        print(f"fuels_daily: {n} rows fetched")
    elif args.command == "fetch-energy-charts":
        import pandas as pd

        from pred_el_prices.pipeline.energy_charts import update_cache

        end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC")
        start = pd.Timestamp(args.start, tz="UTC") if args.start else end - pd.Timedelta(days=14)
        n = update_cache(args.cache_dir, start, end)
        print(f"energy_charts_prices: {n} hourly rows fetched")
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
            evening=args.evening,
            allow_load_surrogate=args.allow_load_surrogate,
            nets_bundle=args.nets_bundle,
        )
    elif args.command == "train-nets":
        import time

        import pandas as pd

        from pred_el_prices.features.dataset import build_dataset
        from pred_el_prices.models.qnn import save_bundle
        from pred_el_prices.production.nets import next_monday, train_ensemble

        monday = (
            pd.Timestamp(args.monday, tz="UTC")
            if args.monday
            else next_monday(pd.Timestamp.now(tz="UTC"))
        )
        t0 = time.perf_counter()
        dataset, _ = build_dataset(args.cache_dir)
        networks, meta = train_ensemble(dataset, monday, n_jobs=args.n_jobs)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        save_bundle(args.out, networks, meta)
        print(
            f"bundle {args.out}: {len(networks)} networks, trained {meta['train_first']}.."
            f"{meta['trained_through']} ({meta['n_train_days']} days), "
            f"{time.perf_counter() - t0:.0f} s"
        )
    elif args.command == "seed-nets-pit":
        import pandas as pd

        from pred_el_prices.production.nets import seed_pit

        seed = seed_pit(
            args.quantiles, args.cal, pd.Timestamp(args.before, tz="UTC"), tuple(args.nets_logs)
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        seed.to_parquet(args.out)
        print(f"PIT seed {args.out}: {len(seed)} hours, {seed.index.min()} .. {seed.index.max()}")
    elif args.command == "backfill-nets":
        import json
        import time

        from pred_el_prices.production.backfill import run as backfill_nets

        t0 = time.perf_counter()
        summary = backfill_nets(
            args.start, args.end, args.cache_dir, args.features, args.out, args.pit_seed,
            args.lear_log, args.lear_history, args.n_jobs,
        )  # fmt: skip
        print(json.dumps(summary["scores"], indent=1))
        print(f"backfill-nets: {time.perf_counter() - t0:.0f} s")
    elif args.command == "build-site":
        from pred_el_prices.production.backfill import build_site

        build_site(args.nets_log, args.lear_log, args.cache_dir, args.out, args.history, args.end)
        n_days = len(list((args.out / "days").glob("*.json")))
        print(f"site JSON written to {args.out} ({n_days} day files)")
    elif args.command == "backfill-history":
        from pred_el_prices.daily_forecast import backfill_history

        backfill_history(args.out, args.runs, args.start, args.end)
    elif args.command == "score-run":
        import json

        import pandas as pd

        from pred_el_prices.eval.scorecard import compare, load_forecast, score

        fc = load_forecast(args.run_dir)
        if args.start or args.end:
            fc = fc.loc[args.start : args.end]
        prices_all = pd.read_parquet(args.dataset)["price_eur_mwh"]
        result = score(fc, prices_all)
        if args.vs_run_dir is not None:
            fc_a = load_forecast(args.vs_run_dir)
            result["vs"] = compare(fc_a, fc)
        if args.start or args.end:
            start_label = args.start or fc.index.min().date().isoformat()
            end_label = args.end or fc.index.max().date().isoformat()
            out_path = args.run_dir / f"scorecard_{start_label}_{end_label}.json"
        else:
            out_path = args.run_dir / "scorecard.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
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
