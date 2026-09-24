"""Quarter-hour shape model: 15-minute prices from hourly percentiles plus a within-hour shape.

The hourly networks forecast each hour's price distribution. This model predicts how the
four quarter-hours deviate from their hourly mean (the deviations sum to zero, so the
hourly means are untouched). Quarter-hour percentiles are the hourly percentiles shifted
by the predicted deviation (comonotone), which the 2026 tests found already calibrated.

Inputs are gate-available only: the TSO 15-minute load forecast (pre-gate), and wind and
solar as HOURLY forecasts linearly interpolated through the hour centres. Production's own
RES forecast is hourly, so the model trains on the TSO hourly values interpolated the same
way (the TSO's own 15-minute detail publishes after the gate and must not be learned from).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def interp_quarters(hourly: pd.Series) -> pd.Series:
    """Hourly means (index = hour start, UTC) -> values at quarter-hour starts.

    Linear through the hour centres (hh:30), evaluated at the quarter centres (hh:07:30,
    ...), so a flat hour stays flat and a ramp spreads across its quarters.
    """
    centres = pd.Series(hourly.to_numpy(), index=hourly.index + pd.Timedelta(minutes=30))
    q_starts = pd.date_range(
        hourly.index.min(), hourly.index.max() + pd.Timedelta(minutes=45), freq="15min"
    )
    q_centres = q_starts + pd.Timedelta(minutes=7.5)
    both = centres.reindex(centres.index.union(q_centres)).interpolate(
        method="time", limit_direction="both"
    )
    return pd.Series(both.reindex(q_centres).to_numpy(), index=q_starts)


def features(
    load15: pd.Series, solar_h: pd.Series, wind_h: pd.Series, p_hour: pd.Series
) -> pd.DataFrame:
    """Shape features at quarter-hour starts (full hours only).

    load15: TSO 15-minute load forecast; solar_h, wind_h: hourly RES forecasts (MW);
    p_hour: the hourly price forecast median (EUR/MWh), index = hour start.
    """
    solar15, wind15 = interp_quarters(solar_h), interp_quarters(wind_h)
    idx = load15.index.intersection(solar15.index).intersection(wind15.index)
    hour = idx.floor("h")
    f = pd.DataFrame(index=idx)
    f["qidx"] = idx.minute // 15
    f["hour_utc"] = idx.hour
    f["dow"] = idx.dayofweek
    for name, s in (("load", load15), ("solar", solar15), ("wind", wind15)):
        v = s.reindex(idx)
        f[f"{name}_dev"] = v - v.groupby(hour).transform("mean")
    load_h = load15.groupby(load15.index.floor("h")).mean()
    hourly = pd.DataFrame({"load": load_h, "solar": solar_h, "wind": wind_h}).dropna()
    hourly["rl"] = hourly["load"] - hourly["solar"] - hourly["wind"]
    ramp = hourly.shift(-1, freq="h") - hourly.shift(1, freq="h")
    for c in ("load", "solar", "rl"):
        f[f"{c}_ramp"] = ramp[c].reindex(hour).to_numpy()
    f["rl_level"] = hourly["rl"].reindex(hour).to_numpy()
    f["p_hour"] = p_hour.reindex(hour).to_numpy()
    f["p_ramp"] = (p_hour.shift(-1, freq="h") - p_hour.shift(1, freq="h")).reindex(hour).to_numpy()
    return f


def target(price15: pd.Series) -> pd.Series:
    """Quarter-hour price minus its hourly mean (full hours only)."""
    hour = price15.index.floor("h")
    full = price15.groupby(hour).transform("size") == 4
    p = price15[full]
    return p - p.groupby(p.index.floor("h")).transform("mean")


def fit_predict(f_train: pd.DataFrame, y_train: pd.Series, f_pred: pd.DataFrame) -> pd.Series:
    """Median shape (absolute-error HGB), centred to sum to zero within each hour."""
    model = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=400, learning_rate=0.05, random_state=0
    )
    model.fit(f_train, y_train.reindex(f_train.index))
    pred = pd.Series(model.predict(f_pred), index=f_pred.index)
    return pred - pred.groupby(pred.index.floor("h")).transform("mean")


def quarter_percentiles(hourly_q: pd.DataFrame, shape: pd.Series, q_cols: list[str]) -> pd.DataFrame:
    """Hourly percentiles (index = hour start) shifted by the shape: quarter-hour percentiles."""
    hq = hourly_q.loc[shape.index.floor("h"), q_cols].to_numpy()
    return pd.DataFrame(
        np.sort(hq + shape.to_numpy()[:, None], axis=1), index=shape.index, columns=q_cols
    )
