"""Build the data-QA report page from the local ENTSO-E cache.

Writes reports/data_qa/: artifact.json (the numbers), PNG figures, and a
self-contained index.html rendering both. No notebooks — this page IS the
exploration output.
"""

import json
from datetime import UTC, datetime
from html import escape
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pred_el_prices.pipeline import cache
from pred_el_prices.pipeline.entsoe import DATASETS, resample_hourly
from pred_el_prices.reporting import qa


def _dataset_names() -> list[str]:
    from pred_el_prices.pipeline.smard import DATASETS as SMARD_DATASETS

    return [f"entsoe/{n}" for n in DATASETS] + [*SMARD_DATASETS, "fuels_daily"]


def compute_artifact(cache_root: Path) -> dict:
    artifact: dict = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "datasets": {},
    }
    for name in _dataset_names():
        df = cache.load(cache_root, name)
        entry: dict = {"rows": len(df)}
        if not df.empty:
            # the hourly-based checks are meaningless for daily settlement data
            is_daily = len(df) > 1 and (df.index[1:] - df.index[:-1]).median() >= pd.Timedelta(
                "1D"
            )
            entry.update(
                {
                    "columns": list(df.columns),
                    "first": str(df.index.min()),
                    "last": str(df.index.max()),
                    "duplicate_timestamps": qa.duplicate_timestamps(df),
                    "resolution_switches": qa.resolution_switches(df),
                }
            )
            if not is_daily:
                entry.update(
                    {
                        "coverage_by_year": qa.coverage_by_year(df),
                        "missing_hours": qa.missing_hours(df),
                        "dst_days": qa.dst_day_check(
                            df, years=sorted({int(y) for y in df.index.year.unique()})
                        ),
                    }
                )
            if "price_eur_mwh" in df.columns:
                entry["price_stats_by_year"] = qa.price_stats_by_year(df["price_eur_mwh"])
        artifact["datasets"][name] = entry
    return artifact


def make_figures(cache_root: Path, out_dir: Path) -> list[str]:
    """Overview figures; returns the written filenames."""
    figures = []
    prices = cache.load(cache_root, "entsoe/day_ahead_prices")
    if prices.empty:
        prices = cache.load(cache_root, "smard_day_ahead_prices")
    if not prices.empty:
        hourly = resample_hourly(prices)["price_eur_mwh"]
        weekly = hourly.resample("7D").agg(["mean", "min", "max"])
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.fill_between(
            weekly.index, weekly["min"], weekly["max"], alpha=0.25, label="weekly min-max"
        )
        ax.plot(weekly.index, weekly["mean"], lw=1.2, label="weekly mean")
        ax.set_ylabel("EUR/MWh")
        ax.set_title("DE-LU day-ahead price (hourly view, weekly aggregates)")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(out_dir / "prices_overview.png", dpi=120)
        plt.close(fig)
        figures.append("prices_overview.png")

        recent = hourly[hourly.index >= hourly.index.max() - pd.Timedelta(days=60)]
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(recent.index, recent.values, lw=0.7)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_ylabel("EUR/MWh")
        ax.set_title("Last 60 days, hourly")
        fig.tight_layout()
        fig.savefig(out_dir / "prices_recent.png", dpi=120)
        plt.close(fig)
        figures.append("prices_recent.png")
    return figures


def _table(rows: list[dict]) -> str:
    if not rows:
        return "<p><em>no data</em></p>"
    cols = list(rows[0].keys())
    head = "".join(f"<th>{escape(str(c))}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>"
        for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html(artifact: dict, figures: list[str]) -> str:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Data QA — ENTSO-E cache</title>",
        (
            "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;"
            "padding:0 1rem;color:#222}table{border-collapse:collapse;margin:1rem 0;"
            "font-size:0.9rem}td,th{border:1px solid #ccc;padding:0.3rem 0.6rem;"
            "text-align:right}th{background:#f0f0f0}img{max-width:100%}"
            "h2{margin-top:2.5rem;border-bottom:2px solid #eee}</style></head><body>"
        ),
        "<h1>Data QA — ENTSO-E cache</h1>",
        f"<p>Generated {escape(artifact['generated_utc'])}. All times UTC.</p>",
    ]
    for fig in figures:
        parts.append(f"<img src='{escape(fig)}' alt='{escape(fig)}'>")
    for name, entry in artifact["datasets"].items():
        parts.append(f"<h2>{escape(name)}</h2>")
        if entry["rows"] == 0:
            parts.append("<p><em>not cached yet</em></p>")
            continue
        parts.append(
            f"<p>{entry['rows']:,} rows, {escape(entry['first'])} → {escape(entry['last'])}, "
            f"{entry['duplicate_timestamps']} duplicate timestamps.</p>"
        )
        parts.append(f"<p>Columns: {escape(', '.join(entry['columns']))}</p>")
        if "coverage_by_year" in entry:
            parts.append("<h3>Coverage by year</h3>")
            parts.append(_table(entry["coverage_by_year"]))
        if "missing_hours" in entry:
            mh = entry["missing_hours"]
            parts.append(f"<h3>Missing hours: {mh['count']}</h3>")
            if mh["count"]:
                shown = ", ".join(mh["hours"])
                suffix = " …(truncated)" if mh.get("truncated") else ""
                parts.append(f"<p style='font-size:0.8rem'>{escape(shown)}{escape(suffix)}</p>")
        if entry["resolution_switches"]:
            parts.append("<h3>Resolution switches</h3>")
            parts.append(_table(entry["resolution_switches"]))
        if entry.get("price_stats_by_year"):
            parts.append("<h3>Price stats by year (native resolution)</h3>")
            parts.append(_table(entry["price_stats_by_year"]))
        if entry.get("dst_days"):
            parts.append("<h3>DST transition days (rows per local Berlin day)</h3>")
            parts.append(_table(entry["dst_days"]))
    parts.append("</body></html>")
    return "".join(parts)


def build_qa_report(cache_root: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = compute_artifact(cache_root)
    (out_dir / "artifact.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    figures = make_figures(cache_root, out_dir)
    page = out_dir / "index.html"
    page.write_text(render_html(artifact, figures), encoding="utf-8")
    return page
