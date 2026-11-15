import argparse
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone

import h5py
import numpy as np
import pandas as pd

from smap_lookup import _lonlat_to_xy

SHORT_NAME = "SPL4SMGP"
VERSION = "008"
VALID_SLOTS = [f"{h:02d}:30" for h in range(1, 24, 3)]
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(os.path.dirname(_HERE), "data")
_DEFAULT_CACHE = os.path.join(_DATA_DIR, "smap_cache.csv")
CACHE_PATH = os.environ.get("SMAP_CACHE", _DEFAULT_CACHE)
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def _target_times(n_weeks, end_date, slot):
    hh, mm = map(int, slot.split(":"))
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(
        hour=hh, minute=mm, tzinfo=timezone.utc)
    return [end - timedelta(weeks=k) for k in range(n_weeks)]


def _login(persist=False):
    import earthaccess
    if os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"):
        earthaccess.login(strategy="environment")
    elif persist:
        earthaccess.login(strategy="all", persist=True)
    else:
        earthaccess.login(strategy="netrc")


def _read_pixel(fobj, lat, lon):
    head = fobj.read(8)
    fobj.seek(0)
    if head != HDF5_MAGIC:
        raise RuntimeError(
            f"The server did not return an HDF5 file (got {head!r}). Check that "
            "'NASA GESDISC DATA ARCHIVE' is authorized in your Earthdata profile.")
    with h5py.File(fobj, "r") as f:
        x0, x1 = f["x"][:2]
        y0, y1 = f["y"][:2]
        ncol, nrow = f["x"].shape[0], f["y"].shape[0]
        px, py = _lonlat_to_xy(lon, lat)
        col = int(np.rint((px - x0) / (x1 - x0))) % ncol
        row = int(np.rint((py - y0) / (y1 - y0)))
        if not 0 <= row < nrow:
            return np.nan, np.nan
        vals = []
        for name in ("sm_surface", "sm_rootzone"):
            v = float(f[f"Geophysical_Data/{name}"][row, col])
            vals.append(np.nan if v < -1000 else v)
        return tuple(vals)


SEARCH_SPAN = timedelta(days=90)


def _find_links(todo):
    import earthaccess
    todo = sorted(todo)
    links = {t: None for t in todo}
    chunks, cur = [], [todo[0]]
    for t in todo[1:]:
        if t - cur[0] <= SEARCH_SPAN:
            cur.append(t)
        else:
            chunks.append(cur)
            cur = [t]
    chunks.append(cur)
    print(f"searching CMR: {len(chunks)} batched search(es) for {len(todo)} timestamps ...", flush=True)
    for chunk in chunks:
        found = earthaccess.search_data(
            short_name=SHORT_NAME, version=VERSION,
            temporal=(chunk[0] - timedelta(minutes=1), chunk[-1] + timedelta(minutes=1)))
        spans = []
        for g in found:
            rng = g["umm"]["TemporalExtent"]["RangeDateTime"]
            b = pd.Timestamp(rng["BeginningDateTime"]).to_pydatetime()
            e = pd.Timestamp(rng["EndingDateTime"]).to_pydatetime()
            h5 = [u for u in g.data_links() if u.endswith(".h5")]
            if h5:
                spans.append((b, e, h5[0]))
        for t in chunk:
            for b, e, url in spans:
                if b <= t <= e:
                    links[t] = url
                    break
    return links


def _fetch_week(args):
    t, url, lat, lon = args
    try:
        import earthaccess
        if url is None:
            return t, "nogranule", np.nan, np.nan
        _login()
        h5 = [url]
        last = None
        for _ in range(3):
            try:
                fobj = earthaccess.open(h5[:1])[0]
                s, r = _read_pixel(fobj, lat, lon)
                return t, "ok", s, r
            except RuntimeError:
                raise
            except Exception as e:
                last = e
        return t, f"error: {last}", np.nan, np.nan
    except Exception as e:
        return t, f"error: {e}", np.nan, np.nan


CACHE_COLS = ["lat", "lon", "time", "sm_surface", "sm_rootzone"]
_OLD_SQLITE = os.path.join(_DATA_DIR, "smap_cache.sqlite")


def _key(lat, lon):
    return round(float(lat), 4), round(float(lon), 4)


def _load_cache():
    if os.path.exists(CACHE_PATH):
        df = pd.read_csv(CACHE_PATH)
    elif os.path.exists(_OLD_SQLITE) and CACHE_PATH == _DEFAULT_CACHE:
        con = sqlite3.connect(_OLD_SQLITE)
        df = pd.read_sql_query("SELECT * FROM obs", con)
        con.close()
        print(f"imported {len(df)} cached rows from the old smap_cache.sqlite")
    else:
        df = pd.DataFrame(columns=CACHE_COLS)
    df = df[CACHE_COLS].copy()
    df["lat"] = df["lat"].astype(float).round(4)
    df["lon"] = df["lon"].astype(float).round(4)
    return df


def _save_cache(df):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    df.sort_values(["lat", "lon", "time"]).to_csv(tmp, index=False)
    os.replace(tmp, CACHE_PATH)


BACKENDS = ("harmony", "earthaccess")
DEFAULT_BACKEND = os.environ.get("SMAP_BACKEND", "harmony")


def weekly_series(lat, lon, n_weeks, end_date=None, slot="22:30", workers=8, use_cache=True,
                  backend=None):
    if slot not in VALID_SLOTS:
        raise ValueError(f"slot must be one of {VALID_SLOTS}")
    if end_date is None:
        end_date = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%d")
    return fetch_times(_target_times(n_weeks, end_date, slot), lat, lon, workers, use_cache, backend)


def fetch_times(times, lat, lon, workers=8, use_cache=True, backend=None):
    backend = backend or DEFAULT_BACKEND
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if backend == "harmony":
        import smap_harmony
        return smap_harmony.fetch_times(times, lat, lon, workers, use_cache)
    times = list(times)
    klat, klon = _key(lat, lon)
    results = {}

    cache = _load_cache() if use_cache else None
    if cache is not None:
        mine = cache[(cache["lat"] == klat) & (cache["lon"] == klon)]
        have = {r.time: (r.sm_surface, r.sm_rootzone) for r in mine.itertuples()}
        for t in times:
            if t.isoformat() in have:
                results[t] = have[t.isoformat()]
        print(f"{len(results)}/{len(times)} timestamps found in cache", flush=True)

    todo = [t for t in times if t not in results]
    if todo:
        _login(persist=True)
        links = _find_links(todo)
        print(f"fetching {len(todo)} timestamps with {min(workers, len(todo))} workers ...", flush=True)
        jobs = [(t, links[t], lat, lon) for t in todo]
        with ProcessPoolExecutor(max_workers=min(workers, len(todo))) as ex:
            for i, (t, status, s, r) in enumerate(ex.map(_fetch_week, jobs), 1):
                print(f"[{i}/{len(todo)}] {t:%Y-%m-%d %H:%M}  {status}", flush=True)
                results[t] = (s, r)
                if status == "ok" and cache is not None:
                    new = pd.DataFrame([[klat, klon, t.isoformat(), s, r]], columns=CACHE_COLS)
                    cache = pd.concat([cache[~((cache["lat"] == klat) & (cache["lon"] == klon)
                                               & (cache["time"] == t.isoformat()))], new],
                                      ignore_index=True)
                    _save_cache(cache)
    if cache is not None and not os.path.exists(CACHE_PATH):
        _save_cache(cache)

    rows = [(t, *results.get(t, (np.nan, np.nan))) for t in sorted(times)]
    return pd.DataFrame(rows, columns=["time", "sm_surface", "sm_rootzone"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("lat", type=float)
    p.add_argument("lon", type=float)
    p.add_argument("n_weeks", type=int)
    p.add_argument("--end", default=None, help="most recent date, YYYY-MM-DD")
    p.add_argument("--slot", default="22:30", help=f"UTC 3-h slot, one of {VALID_SLOTS}")
    p.add_argument("--workers", type=int, default=8, help="parallel processes (default 8)")
    p.add_argument("--no-cache", action="store_true", help="ignore and don't write the cache")
    p.add_argument("--backend", default=None, choices=BACKENDS,
                   help="harmony (default): server-side subsets; earthaccess: full-file reads")
    p.add_argument("--csv", default=None, help="also save the table to this CSV file")
    a = p.parse_args()

    try:
        df = weekly_series(a.lat, a.lon, a.n_weeks, a.end, a.slot, a.workers, not a.no_cache, a.backend)
    except EOFError:
        raise SystemExit("LOGIN NEEDED: no saved Earthdata login and no keyboard to ask for one. "
                         "Run `python smap_timeseries.py 40.7 -74.0 3` once in a terminal, or fill in "
                         "the login fields in the app.")
    print(df.to_string(index=False))
    if a.csv:
        df.to_csv(a.csv, index=False)
        print(f"saved {a.csv}")
