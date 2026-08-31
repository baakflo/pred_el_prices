"""German public holidays as a load feature: population-weighted share on holiday.

Load on a federal holiday runs like a Sunday; on a regional one (Corpus
Christi covers ~64% of the population, Reformation Day ~31%) it lands in
between — a fraction carries that, a national boolean cannot. Everything is
deterministic and offline: fixed dates plus offsets from Easter Sunday
(computus, via dateutil which pandas already depends on), so there is no
calendar service to be down exactly when the fallback is needed.

Holidays are Berlin-local calendar days; hourly UTC timestamps are converted
before lookup, so the 22:00/23:00 UTC hours of the previous day correctly
belong to the holiday.
"""

from datetime import date, timedelta

import pandas as pd
from dateutil.easter import easter

# State population shares (2024 census rounding, sums to ~1.0):
# BW .134 BY .158 BE .044 BB .030 HB .008 HH .022 HE .075 MV .019
# NI .095 NRW .214 RP .049 SL .012 SN .048 ST .025 SH .035 TH .025
FEDERAL = 1.0
EPIPHANY = 0.32  # Jan 6: BW + BY + ST
WOMENS_DAY = 0.06  # Mar 8: BE + MV
CORPUS_CHRISTI = 0.64  # Easter+60: BW + BY + HE + NRW + RP + SL
ASSUMPTION = 0.09  # Aug 15: SL + Catholic-majority Bavaria (~half of BY)
CHILDRENS_DAY = 0.03  # Sep 20: TH
REFORMATION = 0.31  # Oct 31: BB + HB + HH + MV + NI + SN + ST + SH + TH
ALL_SAINTS = 0.57  # Nov 1: BW + BY + NRW + RP + SL
REPENTANCE = 0.05  # Wed before Nov 23: SN


def _repentance_day(year: int) -> date:
    """Buss- und Bettag: the Wednesday strictly before Nov 23."""
    d = date(year, 11, 22)
    return d - timedelta(days=(d.weekday() - 2) % 7)


def year_shares(year: int) -> dict[date, float]:
    """All German holidays of `year` with their population share."""
    e = easter(year)
    return {
        date(year, 1, 1): FEDERAL,
        date(year, 1, 6): EPIPHANY,
        date(year, 3, 8): WOMENS_DAY,
        e - timedelta(days=2): FEDERAL,  # Good Friday
        e + timedelta(days=1): FEDERAL,  # Easter Monday
        date(year, 5, 1): FEDERAL,
        e + timedelta(days=39): FEDERAL,  # Ascension
        e + timedelta(days=50): FEDERAL,  # Whit Monday
        e + timedelta(days=60): CORPUS_CHRISTI,
        date(year, 8, 15): ASSUMPTION,
        date(year, 9, 20): CHILDRENS_DAY,
        date(year, 10, 3): FEDERAL,  # German Unity Day
        date(year, 10, 31): REFORMATION,
        date(year, 11, 1): ALL_SAINTS,
        _repentance_day(year): REPENTANCE,
        date(year, 12, 25): FEDERAL,
        date(year, 12, 26): FEDERAL,
    }


def holiday_share(index: pd.DatetimeIndex, day_offset: int = 0) -> pd.Series:
    """Per-timestamp holiday share of the Berlin-local day, shifted by `day_offset`.

    day_offset −1/+1 gives yesterday's/tomorrow's share — holiday eves and
    bridge days have their own load shape.
    """
    local_days = index.tz_convert("Europe/Berlin") + pd.Timedelta(days=day_offset)
    shares: dict[int, dict[date, float]] = {}
    values = []
    for ts in local_days:
        table = shares.setdefault(ts.year, year_shares(ts.year))
        values.append(table.get(ts.date(), 0.0))
    return pd.Series(values, index=index, name=f"holiday_share_{day_offset:+d}")
