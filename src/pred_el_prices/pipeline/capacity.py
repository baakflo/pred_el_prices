"""Installed wind/solar capacity for Germany (energy-charts.info, CC BY 4.0).

Monthly cumulative installed power per technology, sourced by Fraunhofer ISE
from the Marktstammdatenregister. Needed to normalize generation forecasts to
capacity factors: tree models cannot extrapolate, and German solar grows by
~1 GW/month, so a model trained on raw MW would be biased low at the frontier
within weeks. "Solar AC" (inverter power) is the grid-relevant capacity that
TSO feed-in forecasts are bounded by, not the DC panel peak.
"""

from pathlib import Path

import pandas as pd
import requests

API_URL = "https://api.energy-charts.info/installed_power"

# energy-charts production_type name -> our column
TECHNOLOGIES = {
    "Wind onshore": "wind_onshore_capacity_mw",
    "Wind offshore": "wind_offshore_capacity_mw",
    "Solar AC": "solar_capacity_mw",
}

DATASET = "energy_charts/installed_power"


def fetch_monthly() -> pd.DataFrame:
    """Monthly installed capacity in MW, indexed by month start (UTC)."""
    resp = requests.get(
        API_URL,
        params={"country": "de", "time_step": "monthly", "installation_decommission": "false"},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    index = pd.DatetimeIndex(
        [pd.Timestamp(f"{t[3:]}-{t[:2]}-01", tz="UTC") for t in payload["time"]]
    )
    by_name = {p["name"]: p["data"] for p in payload["production_types"]}
    df = pd.DataFrame(
        {
            col: pd.array(by_name[name], dtype="float64") * 1000.0
            for name, col in TECHNOLOGIES.items()
        },
        index=index,
    )
    return df.dropna(how="all")


def update_cache(cache_dir: Path) -> pd.DataFrame:
    from pred_el_prices.pipeline import cache

    df = fetch_monthly()
    cache.upsert(cache_dir, DATASET, df)
    return df


def hourly_capacity(cache_dir: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Capacity interpolated to an hourly index (linear between month points).

    The tail beyond the last month point is extended flat — in production the
    current month's value is the best available estimate.
    """
    from pred_el_prices.pipeline import cache

    monthly = cache.load(cache_dir, DATASET)
    if monthly.empty:
        raise RuntimeError(f"no cached capacity; run `pep fetch-capacity` (cache: {cache_dir})")
    combined = monthly.reindex(monthly.index.union(index))
    combined = combined.interpolate(method="time").ffill()
    return combined.reindex(index)
