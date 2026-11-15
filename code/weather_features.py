import argparse
import contextlib
import io
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "Weather data.py"))
from APIs.NASApowerAPI import fetch_nasa_data  # noqa: E402

CACHE_PATH = os.environ.get("WEATHER_CACHE", os.path.join(ROOT, "data", "weather_cache.csv"))
POWER_PARAMS = ["PRECTOTCORR", "T2M", "RH2M", "WS10M"]
WEATHER_FEATURES = ["rain_7d", "rain_28d", "t2m_7d", "rh2m_7d", "wind_7d"]
CACHE_COLS = ["lat", "lon", "time"] + WEATHER_FEATURES


def _weekly_features(hourly, times):
    h = hourly.set_index("Timestamp").sort_index()
    h = h[~h.index.duplicated()]

    def roll(col, window, how, min_frac):
        hours = pd.Timedelta(window).total_seconds() / 3600
        r = getattr(h[col].rolling(window, min_periods=int(hours * min_frac)), how)()
        return r.reindex(times, method="ffill", tolerance=pd.Timedelta("2h"))

    return pd.DataFrame({
        "rain_7d": roll("PRECTOTCORR", "7D", "sum", 0.8),
        "rain_28d": roll("PRECTOTCORR", "28D", "sum", 0.8),
        "t2m_7d": roll("T2M", "7D", "mean", 0.8),
        "rh2m_7d": roll("RH2M", "7D", "mean", 0.8),
        "wind_7d": roll("WS10M", "7D", "mean", 0.8),
    })


POWER_MAX_DAYS = 730


def _fetch_hourly(lat, lon, start, end):
    parts, s = [], start
    while s <= end:
        e = min(s + timedelta(days=POWER_MAX_DAYS - 1), end)
        parts.append(fetch_nasa_data(lat, lon, s, e, POWER_PARAMS, time_standard="UTC"))
        s = e + timedelta(days=1)
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates("Timestamp").sort_values("Timestamp").reset_index(drop=True)


def add_weather(df, lat, lon):
    klat, klon = round(float(lat), 4), round(float(lon), 4)
    cache = (pd.read_csv(CACHE_PATH) if os.path.exists(CACHE_PATH)
             else pd.DataFrame(columns=CACHE_COLS))
    cache["lat"], cache["lon"] = cache["lat"].astype(float).round(4), cache["lon"].astype(float).round(4)
    mine = cache[(cache["lat"] == klat) & (cache["lon"] == klon)].set_index("time")

    keys = df["time"].map(lambda t: t.isoformat())
    missing = df.loc[~keys.isin(mine.index), "time"]
    if len(missing):
        t_naive = pd.DatetimeIndex(missing).tz_convert("UTC").tz_localize(None)
        start = (t_naive.min() - timedelta(days=30)).date()
        end = t_naive.max().date()
        print(f"fetching NASA POWER hourly weather {start} .. {end} "
              f"for {len(missing)} weeks (cached: {len(df) - len(missing)})", flush=True)
        hourly = _fetch_hourly(klat, klon, start, end)
        print(f"  got {len(hourly)} hourly rows", flush=True)
        new = _weekly_features(hourly, t_naive)
        new.insert(0, "time", [t.isoformat() for t in missing])
        new.insert(0, "lon", klon)
        new.insert(0, "lat", klat)
        complete = new[WEATHER_FEATURES].notna().all(axis=1)
        merged = pd.concat([cache[~((cache["lat"] == klat) & (cache["lon"] == klon)
                                    & cache["time"].isin(new.loc[complete, "time"]))],
                            new[complete]], ignore_index=True)
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        merged[CACHE_COLS].to_csv(CACHE_PATH, index=False)
        mine = pd.concat([mine, new[complete].set_index("time")])
        mine = mine[~mine.index.duplicated(keep="last")]
        extra = new.set_index("time")
        mine = pd.concat([mine, extra[~extra.index.isin(mine.index)]])
    out = df.copy()
    for c in WEATHER_FEATURES:
        out[c] = keys.map(mine[c]).astype(float).to_numpy()
    return out


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


WEATHER_UNITS = {"rain_7d": "mm", "rain_28d": "mm", "t2m_7d": "°C", "rh2m_7d": "%", "wind_7d": "m/s"}
WEATHER_BOUNDS = {"rain_7d": (0, None), "rain_28d": (0, None), "t2m_7d": (None, None),
                  "rh2m_7d": (0, 100), "wind_7d": (0, None)}


def forecast_weather(df, horizon, targets, inputs, forecast_fn=None, offsets=None):
    import xgboost_model as xm
    forecast_fn = forecast_fn or xm.forecast
    old = xm.TARGETS
    try:
        xm.TARGETS = list(targets)
        res = forecast_fn(df, horizon=horizon, features=list(inputs))
    finally:
        xm.TARGETS = old
    fc = res["forecast"].copy()
    for k in targets:
        if offsets:
            fc[k] = fc[k] + float(offsets.get(k, 0.0))
        lo, hi = WEATHER_BOUNDS.get(k, (None, None))
        fc[k] = fc[k].clip(lower=lo, upper=hi)
    res["forecast"] = fc
    return res


if __name__ == "__main__":
    from smap_timeseries import weekly_series
    import xgboost_model as xm

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("lat", type=float)
    p.add_argument("lon", type=float)
    p.add_argument("n_weeks", type=int)
    p.add_argument("--end", default=None)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()

    df = weekly_series(a.lat, a.lon, a.n_weeks, a.end, "22:30", a.workers)
    df = add_weather(df, a.lat, a.lon)
    print(f"\n{len(df)} weeks; weather coverage: "
          f"{df[WEATHER_FEATURES].notna().all(axis=1).sum()} complete weeks")
    print(df.tail(5).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    base = _quiet(xm.forecast, df)
    with_w = _quiet(xm.forecast, df, features=xm.TARGETS + WEATHER_FEATURES)

    m = base["metrics"].merge(with_w["metrics"], on=["target", "weeks_ahead"], suffixes=("_soil_only", "_with_weather"))
    show = m[["target", "weeks_ahead", "scored_rows_soil_only", "rmse_soil_only", "rmse_with_weather"]].copy()
    show.columns = ["target", "weeks_ahead", "rows", "RMSE soil-only", "RMSE +weather"]
    print("\n=== Does weather help? (validation, lower RMSE is better) ===")
    print(show.round(4).to_string(index=False))
    better = (m["rmse_with_weather"] < m["rmse_soil_only"]).sum()
    print(f"\nWeather lowered the error in {better} of {len(m)} target/horizon cases; "
          f"mean RMSE {m['rmse_soil_only'].mean():.4f} -> {m['rmse_with_weather'].mean():.4f}")
    print("\nTop features with weather (surface, 1 week):")
    print(with_w["importance"].round(3).to_string())
