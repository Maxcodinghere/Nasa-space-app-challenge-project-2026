import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


_R = 6378137.0
_E2 = 0.00669437999014
_E = np.sqrt(_E2)
_LAT_TS = np.deg2rad(30.0)


def _q(sin_phi):
    return (1 - _E2) * (sin_phi / (1 - _E2 * sin_phi**2)
                        - (1 / (2 * _E)) * np.log((1 - _E * sin_phi) / (1 + _E * sin_phi)))


_K0 = np.cos(_LAT_TS) / np.sqrt(1 - _E2 * np.sin(_LAT_TS) ** 2)


def _lonlat_to_xy(lon, lat):
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    x = _R * _K0 * np.deg2rad(lon)
    y = _R * _q(np.sin(np.deg2rad(lat))) / (2 * _K0)
    return x, y


VALID_INTERVALS = [3, 6, 12, 24]
MAX_REQUESTS = 400


def _slot_times(start, end, interval):
    step = interval // 3
    out = []
    day = pd.Timestamp(start).normalize()
    while day <= pd.Timestamp(end):
        for k in range(8):
            if k % step == 0:
                out.append((day + pd.Timedelta(hours=1.5 + 3 * k)).tz_localize("UTC"))
        day += pd.Timedelta(days=1)
    return out


def fetch_smap_data(latitude, longitude, start_date, end_date, interval=3,
                    workers=8, use_cache=True, max_requests=MAX_REQUESTS):
    if not -90 <= latitude <= 90:
        raise ValueError("Invalid latitude.")
    if not -180 <= longitude <= 180:
        raise ValueError("Invalid longitude.")
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Invalid date.")
    if start > end:
        raise ValueError("Invalid date range.")
    if start < pd.Timestamp("2015-03-31"):
        raise ValueError("SMAP L4 data starts on 2015-03-31.")
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Invalid sampling interval; choose one of {VALID_INTERVALS}.")

    times = _slot_times(start, end, interval)
    if len(times) > max_requests:
        raise ValueError(f"{len(times)} observations requested (limit {max_requests}); "
                         "shorten the date range or use a larger interval.")

    import smap_timeseries as sts
    df = sts.fetch_times(times, latitude, longitude, workers, use_cache)

    out = pd.DataFrame({
        "Timestamp": df["time"].dt.tz_convert("UTC").dt.tz_localize(None),
        "Latitude": latitude,
        "Longitude": longitude,
        "sm_surface": df["sm_surface"],
        "sm_rootzone": df["sm_rootzone"],
    })
    return out.reset_index(drop=True)


def get_soil_moisture(lat, lon, when=None):
    t = pd.Timestamp(when) if when is not None else pd.Timestamp(
        datetime.now(timezone.utc) - timedelta(days=4)).tz_localize(None)
    t = t.tz_localize(None) if t.tzinfo else t
    day = t.normalize()
    cand = _slot_times(day, day, 3)
    slot = min(cand, key=lambda c: abs(c.tz_localize(None) - t))
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("Invalid latitude/longitude.")
    import smap_timeseries as sts
    row = sts.fetch_times([slot], lat, lon).iloc[0]
    return {"time": slot.tz_localize(None), "sm_surface": float(row["sm_surface"]),
            "sm_rootzone": float(row["sm_rootzone"])}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("lat", type=float)
    p.add_argument("lon", type=float)
    p.add_argument("--start", default=None, help="YYYY-MM-DD (default: single snapshot)")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (default = start)")
    p.add_argument("--interval", type=int, default=3, help=f"hours between observations {VALID_INTERVALS}")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--csv", default=None)
    a = p.parse_args()

    try:
        if a.start is None:
            r = get_soil_moisture(a.lat, a.lon)
            print(f"Location: {a.lat}, {a.lon}   time: {r['time']} UTC")
            print(f"Surface soil moisture:   {r['sm_surface']:.4f} m3/m3")
            print(f"Root-zone soil moisture: {r['sm_rootzone']:.4f} m3/m3")
        else:
            df = fetch_smap_data(a.lat, a.lon, a.start, a.end or a.start, a.interval, a.workers)
            print(df.to_string(index=False))
            if a.csv:
                df.to_csv(a.csv, index=False)
                print(f"saved {a.csv}")
    except EOFError:
        raise SystemExit("LOGIN NEEDED: run `python smap_timeseries.py 40.7 -74.0 3` once in a "
                         "terminal to save your Earthdata login.")
