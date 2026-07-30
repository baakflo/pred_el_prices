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

    args = parser.parse_args()
    if args.command == "archive-weather":
        from pred_el_prices.pipeline.dwd import archive_run

        archive_run(args.date, args.archive_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
