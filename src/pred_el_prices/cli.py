"""Command-line entry point. Target (Phase 0 deliverable): one command produces a clean
feature/target table from raw APIs, e.g. `pep build-dataset`."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="pep", description=__doc__)
    parser.add_subparsers(dest="command")
    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()
