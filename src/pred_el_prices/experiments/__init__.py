"""Named, reproducible experiments: `pep run <name> [--set key=value ...]`.

Every experiment is a function taking keyword params and an output directory;
it writes its artifacts (forecasts, metrics.json, config.json) there and
returns the metrics dict. Artifact layout: runs/<name>-<UTC timestamp>/.
"""

import json
from datetime import UTC, datetime
from pathlib import Path


def run(name: str, params: dict, runs_root: Path = Path("runs")) -> dict:
    from pred_el_prices.experiments import lear_de, load_de, res_de

    registry = {
        "lear-de": lear_de.run,
        "res-de": res_de.run,
        "load-de": load_de.run,
    }
    if name not in registry:
        raise SystemExit(f"unknown experiment {name!r}; available: {sorted(registry)}")

    out_dir = runs_root / f"{name}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"experiment": name, **params}, indent=2))

    metrics = registry[name](out_dir=out_dir, **params)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"artifacts: {out_dir}")
    print(json.dumps(metrics, indent=2))
    return metrics
