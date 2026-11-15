import contextlib
import glob
import netrc
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import h5py
import numpy as np
import pandas as pd
import requests

from smap_lookup import _lonlat_to_xy

PRODUCT = "SPL4SMGP"
VERSION = "008"
COLLECTION_ID = "C3480440870-NSIDC_CPRD"
VARIABLES = ["Geophysical_Data/sm_surface", "Geophysical_Data/sm_rootzone"]
CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"

NCOL, NROW = 3856, 1624
CELL_M = 36032220.84 / 4000
_XUL, _YUL = -17367530.44516138, 7314540.79258289
_XC0, _YC0 = _XUL + CELL_M / 2, _YUL - CELL_M / 2
GRID_TOL = 1e-2

SEARCH_SPAN = timedelta(days=90)
JOB_CHUNK = int(os.environ.get("SMAP_HARMONY_CHUNK", 50))
JOB_TIMEOUT = float(os.environ.get("SMAP_HARMONY_TIMEOUT", 1800))
DOWNLOAD_THREADS = int(os.environ.get("SMAP_DOWNLOAD_THREADS", 16))
POLL_SECONDS = 5.0
RETRIES = 3

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(os.path.dirname(_HERE), "data")
CELL_CACHE = os.environ.get("SMAP_CELL_CACHE", os.path.join(_DATA_DIR, "smap_cell_cache.csv"))
CELL_MAP = os.environ.get("SMAP_CELL_MAP", os.path.join(os.path.dirname(CELL_CACHE), "smap_cell_map.csv"))
LEGACY_CACHE = os.environ.get("SMAP_CACHE", os.path.join(_DATA_DIR, "smap_cache.csv"))

OBS_COLS = ["product", "version", "row", "col", "time", "sm_surface", "sm_rootzone"]
MAP_COLS = ["lat", "lon", "row", "col", "cell_lat", "cell_lon", "verified"]
_FILE_TIME = re.compile(r"_(\d{8}T\d{6})_")


class NoLandData(Exception):
    pass


def cell_for(lat, lon):
    px, py = _lonlat_to_xy(lon, lat)
    col = int(np.rint((px - _XC0) / CELL_M)) % NCOL
    row = int(np.rint((_YC0 - py) / CELL_M))
    if not 0 <= row < NROW:
        raise NoLandData(f"latitude {lat} is outside the SMAP L4 grid (about +-85.04 deg)")
    return row, col


def _bbox(lat, lon):
    mlon = 0.1
    mlat = max(0.1, 2.0 * 0.0405 / max(np.cos(np.deg2rad(lat)), 0.05))
    return (max(-180.0, lon - mlon), max(-90.0, lat - mlat),
            min(180.0, lon + mlon), min(90.0, lat + mlat))


@contextlib.contextmanager
def _file_lock(path, timeout=900.0, stale=7200.0):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    deadline = time.time() + timeout
    while True:
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(path) > stale:
                    os.remove(path)
                    continue
            except OSError:
                pass
            if time.time() > deadline:
                raise TimeoutError(f"timed out waiting for lock {path}")
            time.sleep(0.5)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)


def _atomic_csv(df, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _iso(t):
    t = pd.Timestamp(t)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.isoformat()


def _read_obs():
    if os.path.exists(CELL_CACHE):
        df = pd.read_csv(CELL_CACHE, dtype={"product": str, "version": str})
        df["version"] = df["version"].str.zfill(3)
        return df[OBS_COLS]
    return pd.DataFrame(columns=OBS_COLS)


def _read_map():
    if os.path.exists(CELL_MAP):
        return pd.read_csv(CELL_MAP)[MAP_COLS]
    return pd.DataFrame(columns=MAP_COLS)


def _import_legacy():
    if os.path.exists(CELL_CACHE) or not os.path.exists(LEGACY_CACHE):
        return
    with _file_lock(CELL_CACHE + ".lock"):
        if os.path.exists(CELL_CACHE):
            return
        old = pd.read_csv(LEGACY_CACHE)
        rows = []
        for (lat, lon), g in old.groupby(["lat", "lon"]):
            try:
                row, col = cell_for(lat, lon)
            except NoLandData:
                continue
            rows.append(pd.DataFrame({"product": PRODUCT, "version": VERSION, "row": row, "col": col,
                                      "time": g["time"].values, "sm_surface": g["sm_surface"].values,
                                      "sm_rootzone": g["sm_rootzone"].values}))
        new = (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=OBS_COLS))
        new = new.drop_duplicates(["product", "version", "row", "col", "time"], keep="last")
        _atomic_csv(new[OBS_COLS], CELL_CACHE)
        print(f"imported {len(new)} cached rows from the old smap_cache.csv", flush=True)


def _cached(row, col, times):
    obs = _read_obs()
    obs = obs[(obs["product"] == PRODUCT) & (obs["version"] == VERSION)
              & (obs["row"] == row) & (obs["col"] == col)]
    want = {_iso(t) for t in times}
    obs = obs[obs["time"].isin(want)]
    return {r.time: (r.sm_surface, r.sm_rootzone) for r in obs.itertuples()}


def _store(row, col, new_rows):
    if not new_rows:
        return
    add = pd.DataFrame([[PRODUCT, VERSION, row, col, k, s, r] for k, (s, r) in new_rows.items()],
                       columns=OBS_COLS)
    with _file_lock(CELL_CACHE + ".lock"):
        cur = _read_obs()
        out = pd.concat([cur, add], ignore_index=True).drop_duplicates(
            ["product", "version", "row", "col", "time"], keep="last")
        _atomic_csv(out.sort_values(["row", "col", "time"]), CELL_CACHE)


def _store_map(lat, lon, row, col, cell_lat=np.nan, cell_lon=np.nan, verified=False):
    lat, lon = round(float(lat), 4), round(float(lon), 4)
    add = pd.DataFrame([[lat, lon, row, col, cell_lat, cell_lon, verified]], columns=MAP_COLS)
    with _file_lock(CELL_MAP + ".lock"):
        cur = _read_map()
        keep = cur[~((cur["lat"] == lat) & (cur["lon"] == lon))]
        _atomic_csv(pd.concat([keep, add], ignore_index=True), CELL_MAP)


def _mapped_cell(lat, lon):
    m = _read_map()
    hit = m[(m["lat"] == round(float(lat), 4)) & (m["lon"] == round(float(lon), 4))]
    if len(hit):
        return int(hit.iloc[-1]["row"]), int(hit.iloc[-1]["col"])
    row, col = cell_for(lat, lon)
    _store_map(lat, lon, row, col)
    return row, col


def _windows(times):
    times = sorted(times)
    out, cur = [], [times[0]]
    for t in times[1:]:
        if t - cur[0] <= SEARCH_SPAN:
            cur.append(t)
        else:
            out.append(cur)
            cur = [t]
    out.append(cur)
    return out


def find_granules(times):
    found = {t: None for t in times}
    for chunk in _windows(times):
        start = (chunk[0] - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        stop = (chunk[-1] + timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        entries, page = [], 1
        while True:
            r = requests.get(CMR_GRANULES, params={
                "collection_concept_id": COLLECTION_ID, "temporal": f"{start},{stop}",
                "page_size": 2000, "page_num": page, "sort_key": "start_date"}, timeout=60)
            r.raise_for_status()
            got = r.json()["feed"]["entry"]
            entries += got
            if len(got) < 2000:
                break
            page += 1
        spans = [(pd.Timestamp(e["time_start"]).to_pydatetime(), pd.Timestamp(e["time_end"]).to_pydatetime(),
                  e["id"]) for e in entries]
        for t in chunk:
            for b, e, gid in spans:
                if b <= t <= e:
                    found[t] = gid
                    break
    return found


def _credentials():
    for u, p in (("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"), ("EDL_USERNAME", "EDL_PASSWORD")):
        if os.environ.get(u) and os.environ.get(p):
            return os.environ[u], os.environ[p]
    for name in (".netrc", "_netrc"):
        path = os.path.join(os.path.expanduser("~"), name)
        if os.path.exists(path):
            try:
                a = netrc.netrc(path).authenticators("urs.earthdata.nasa.gov")
            except (netrc.NetrcParseError, OSError):
                a = None
            if a:
                return a[0], a[2]
    raise EOFError("No Earthdata login found: set EARTHDATA_USERNAME / EARTHDATA_PASSWORD or save it "
                   "in ~/.netrc (Windows: ~/_netrc).")


def _client():
    from harmony import Client
    return Client(auth=_credentials())


def _download_all(client, job, outdir):
    from urllib.parse import urlparse
    urls = list(client.result_urls(job))
    if not urls:
        return []
    session = client._session()

    def one(url):
        path = os.path.join(outdir, os.path.basename(urlparse(url).path))
        last = None
        for attempt in range(1, RETRIES + 1):
            try:
                with session.get(url, timeout=120, stream=True) as r:
                    r.raise_for_status()
                    with open(path + ".part", "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                os.replace(path + ".part", path)
                return path
            except Exception as e:
                last = e
                time.sleep(2 * attempt)
        raise RuntimeError(f"download failed for {os.path.basename(url)}: {last}")

    files = [one(urls[0])]
    with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as ex:
        files += list(ex.map(one, urls[1:]))
    return files


def _run_job(client, granule_ids, bbox, outdir, log):
    from harmony import BBox, Collection, Request
    req = Request(collection=Collection(id=COLLECTION_ID), spatial=BBox(*bbox),
                  granule_id=list(granule_ids), variables=VARIABLES,
                  format="application/netcdf", skip_preview=True)
    tm = {"submit": 0.0, "wait": 0.0, "transfer": 0.0}
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            t0 = time.time()
            job = client.submit(req)
            tm["submit"] += time.time() - t0
            log(f"  Harmony job {job[:8]} submitted ({len(granule_ids)} granules)")
            t0, shown = time.time(), None
            while True:
                st = client.status(job)
                now = (st["status"], st.get("progress", 0))
                if now != shown:
                    shown = now
                    log(f"  job {job[:8]}: {now[0]} {now[1]}%")
                if st["status"] in ("successful", "complete_with_errors"):
                    break
                if st["status"] in ("failed", "canceled"):
                    raise RuntimeError(f"Harmony job {st['status']}: {st.get('message')}")
                if time.time() - t0 > JOB_TIMEOUT:
                    with contextlib.suppress(Exception):
                        client.cancel(job)
                    raise TimeoutError(f"Harmony job {job[:8]} exceeded {JOB_TIMEOUT:.0f}s")
                time.sleep(POLL_SECONDS)
            tm["wait"] += time.time() - t0
            if st["status"] == "complete_with_errors":
                log(f"  job {job[:8]} completed WITH ERRORS: {st.get('message')}")
            t0 = time.time()
            files = _download_all(client, job, outdir)
            tm["transfer"] += time.time() - t0
            return files, tm
        except TimeoutError:
            raise
        except Exception as e:
            last = e
            log(f"  attempt {attempt}/{RETRIES} failed: {type(e).__name__}: {e}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"Harmony request failed after {RETRIES} attempts: {last}")


def _file_time(path):
    m = _FILE_TIME.search(os.path.basename(path))
    if not m:
        raise RuntimeError(f"cannot read the timestamp from file name {os.path.basename(path)}")
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def read_cell(path, row, col):
    with h5py.File(path, "r") as f:
        x, y = f["x"][:], f["y"][:]
        cols, rows = (x - _XC0) / CELL_M, (_YC0 - y) / CELL_M
        ci, ri = np.rint(cols), np.rint(rows)
        if np.abs(cols - ci).max() > GRID_TOL or np.abs(rows - ri).max() > GRID_TOL:
            raise RuntimeError("subset coordinates are not on the expected EASE-Grid 2.0 9 km grid")
        ci = ci.astype(int) % NCOL
        ri = ri.astype(int)
        jj, ii = np.where(ci == col)[0], np.where(ri == row)[0]
        if not len(jj) or not len(ii):
            raise RuntimeError(f"cell (row {row}, col {col}) is not in the returned subset "
                               f"(rows {ri.min()}-{ri.max()}, cols {ci.min()}-{ci.max()})")
        i, j = int(ii[0]), int(jj[0])
        vals = []
        for name in VARIABLES:
            d = f[name]
            v = float(d[i, j])
            fill = d.attrs.get("_FillValue")
            lo, hi = d.attrs.get("valid_min"), d.attrs.get("valid_max")
            bad = (fill is not None and np.isclose(v, float(np.ravel(fill)[0]))) or not np.isfinite(v)
            bad = bad or (lo is not None and v < float(np.ravel(lo)[0])) \
                      or (hi is not None and v > float(np.ravel(hi)[0]))
            vals.append(np.nan if bad else v)
        clat = float(f["cell_lat"][i, j]) if "cell_lat" in f else np.nan
        clon = float(f["cell_lon"][i, j]) if "cell_lon" in f else np.nan
    return vals[0], vals[1], clat, clon


def fetch_times(times, lat, lon, workers=4, use_cache=True, log=None):
    log = log or (lambda s: print(s, flush=True))
    times = list(times)
    t_start = time.time()
    timing = {"cmr": 0.0, "submit": 0.0, "wait": 0.0, "transfer": 0.0, "extract": 0.0}
    nbytes = 0

    row, col = _mapped_cell(lat, lon) if use_cache else cell_for(lat, lon)
    log(f"cell for ({lat}, {lon}): row {row}, col {col}")
    results = {}
    if use_cache:
        _import_legacy()
        have = _cached(row, col, times)
        results = {t: have[_iso(t)] for t in times if _iso(t) in have}
        log(f"{len(results)}/{len(times)} timestamps found in cache")

    todo = [t for t in times if t not in results]
    if todo:
        lock = _file_lock(f"{CELL_CACHE}.cell{row}_{col}.fetch.lock", timeout=3600) if use_cache \
            else contextlib.nullcontext()
        with lock:
            if use_cache:
                have = _cached(row, col, todo)
                for t in list(todo):
                    if _iso(t) in have:
                        results[t] = have[_iso(t)]
                todo = [t for t in todo if t not in results]
            if todo:
                _fetch_missing(todo, lat, lon, row, col, results, timing, workers, use_cache, log)
        nbytes = timing.pop("bytes", 0)

    rows = [(t, *results.get(t, (np.nan, np.nan))) for t in sorted(times)]
    out = pd.DataFrame(rows, columns=["time", "sm_surface", "sm_rootzone"])
    got = out[["sm_surface", "sm_rootzone"]].notna().any(axis=1).sum()
    if len(out) and not got:
        log(f"WARNING: no valid soil-moisture values for cell (row {row}, col {col}) - "
            "ocean, ice or no published granules.")
    if todo:
        log(f"timing: CMR {timing['cmr']:.1f}s | submit {timing['submit']:.1f}s | "
            f"Harmony wait (queue+processing) {timing['wait']:.1f}s | transfer {timing['transfer']:.1f}s | "
            f"extract {timing['extract']:.2f}s | total {time.time() - t_start:.1f}s | "
            f"downloaded {nbytes / 1e6:.2f} MB")
    return out


def _fetch_missing(todo, lat, lon, row, col, results, timing, workers, use_cache, log):
    t0 = time.time()
    gids = find_granules(todo)
    timing["cmr"] += time.time() - t0
    missing = [t for t, g in gids.items() if g is None]
    for t in missing:
        results[t] = (np.nan, np.nan)
        log(f"  {t:%Y-%m-%d %H:%M}  nogranule")
    by_name = {t: g for t, g in gids.items() if g}
    if not by_name:
        return
    order = sorted(by_name)
    chunks = [order[i:i + JOB_CHUNK] for i in range(0, len(order), JOB_CHUNK)]
    bbox = _bbox(lat, lon)
    client = _client()
    tmp = tempfile.mkdtemp(prefix="smap_harmony_")
    new_rows, verified = {}, None
    try:
        def run(i_chunk):
            i, chunk = i_chunk
            d = os.path.join(tmp, f"job{i}")
            os.makedirs(d)
            files, tm = _run_job(client, [by_name[t] for t in chunk], bbox, d, log)
            return chunk, files, tm
        log(f"Harmony: {len(order)} granules in {len(chunks)} job(s), bbox {tuple(round(b, 3) for b in bbox)}")
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(chunks)))) as ex:
            for chunk, files, tm in ex.map(run, enumerate(chunks)):
                for k in ("submit", "wait", "transfer"):
                    timing[k] += tm[k]
                timing["bytes"] = timing.get("bytes", 0) + sum(os.path.getsize(f) for f in files)
                t0 = time.time()
                by_time = {}
                for f in files:
                    try:
                        by_time[_file_time(f)] = f
                    except RuntimeError as e:
                        log(f"  skipped file: {e}")
                for t in chunk:
                    f = by_time.get(t)
                    if f is None:
                        results[t] = (np.nan, np.nan)
                        log(f"  {t:%Y-%m-%d %H:%M}  error: no file returned for this granule")
                        continue
                    try:
                        s, r, clat, clon = read_cell(f, row, col)
                    except Exception as e:
                        results[t] = (np.nan, np.nan)
                        log(f"  {t:%Y-%m-%d %H:%M}  error: {e}")
                        continue
                    if verified is None and np.isfinite(clat):
                        verified = (clat, clon)
                    results[t] = (s, r)
                    new_rows[_iso(t)] = (s, r)
                    log(f"  {t:%Y-%m-%d %H:%M}  ok")
                timing["extract"] += time.time() - t0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if use_cache:
        _store(row, col, new_rows)
        if verified:
            _store_map(lat, lon, row, col, verified[0], verified[1], True)
