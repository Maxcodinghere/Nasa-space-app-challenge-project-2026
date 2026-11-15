import os
import sys
from datetime import datetime, timedelta, timezone

import h5py
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import smap_harmony as sh  # noqa: E402

LAT, LON = 40.7, -74.0
UTC = timezone.utc


def T(day, hour=22, minute=30):
    return datetime(2026, 10, day, hour, minute, tzinfo=UTC)


def write_subset(path, row, col, fill_target=False, off_grid=False, span=2):
    cols = np.arange(col - span + 1, col + span + 1)
    rows = np.arange(row - span + 1, row + span + 1)
    x = sh._XC0 + cols * sh.CELL_M + (123.0 if off_grid else 0.0)
    y = sh._YC0 - rows * sh.CELL_M
    ii, jj = np.meshgrid(np.arange(len(rows)), np.arange(len(cols)), indexing="ij")
    surf = (0.1 + 0.01 * ii + 0.001 * jj).astype("f4")
    root = (surf + 0.05).astype("f4")
    ti, tj = int(np.where(rows == row)[0][0]), int(np.where(cols == col)[0][0])
    if fill_target:
        surf[ti, tj] = root[ti, tj] = -9999.0
    with h5py.File(path, "w") as f:
        f["x"], f["y"] = x, y
        g = f.create_group("Geophysical_Data")
        for name, arr in (("sm_surface", surf), ("sm_rootzone", root)):
            d = g.create_dataset(name, data=arr)
            d.attrs["_FillValue"] = np.float32(-9999.0)
            d.attrs["valid_min"], d.attrs["valid_max"] = np.float32(0.0), np.float32(0.9)
        f["cell_lat"] = np.full(arr.shape, 40.73, "f4")
        f["cell_lon"] = np.full(arr.shape, -73.99, "f4")
    return float(surf[ti, tj]), float(root[ti, tj])


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(sh, "CELL_CACHE", str(tmp_path / "smap_cell_cache.csv"))
    monkeypatch.setattr(sh, "CELL_MAP", str(tmp_path / "smap_cell_map.csv"))
    monkeypatch.setattr(sh, "LEGACY_CACHE", str(tmp_path / "legacy.csv"))

    class Fake:
        jobs = []
        no_granule = set()
        fill_target = False
    fake = Fake()
    fake.jobs = []
    fake.no_granule = set()

    def find_granules(times):
        return {t: (None if t in fake.no_granule else f"G-{t.strftime('%Y%m%dT%H%M%S')}") for t in times}

    def run_job(client, gids, bbox, outdir, log):
        fake.jobs.append(list(gids))
        row, col = sh.cell_for(LAT, LON)
        files = []
        for g in gids:
            p = os.path.join(outdir, f"123_SMAP_L4_SM_gph_{g[2:]}_Vv8013_001_subsetted.nc4")
            write_subset(p, row, col, fill_target=fake.fill_target)
            files.append(p)
        return files, {"submit": 0.0, "wait": 0.0, "transfer": 0.0}

    monkeypatch.setattr(sh, "find_granules", find_granules)
    monkeypatch.setattr(sh, "_run_job", run_job)
    monkeypatch.setattr(sh, "_client", lambda: object())
    return fake


quiet = dict(log=lambda s: None)


def test_cell_for_is_consistent_and_known():
    assert sh.cell_for(LAT, LON) == (281, 1135)
    assert sh.cell_for(LAT + 0.001, LON + 0.001) == sh.cell_for(LAT, LON)
    assert sh.cell_for(LAT, LON + 0.5) != sh.cell_for(LAT, LON)
    assert sh.cell_for(0.0, 180.0)[1] in (0, sh.NCOL - 1)


def test_cell_for_outside_grid_is_explicit():
    with pytest.raises(sh.NoLandData):
        sh.cell_for(89.9, 0.0)


def test_read_cell_picks_the_requested_cell_not_a_neighbour(tmp_path):
    row, col = 281, 1135
    p = str(tmp_path / "a.nc4")
    exp_s, exp_r = write_subset(p, row, col)
    s, r, clat, clon = sh.read_cell(p, row, col)
    assert (s, r) == pytest.approx((exp_s, exp_r))
    s2, _, _, _ = sh.read_cell(p, row, col + 1)
    assert s2 != pytest.approx(s)


def test_read_cell_fill_value_becomes_nan(tmp_path):
    p = str(tmp_path / "a.nc4")
    write_subset(p, 281, 1135, fill_target=True)
    s, r, *_ = sh.read_cell(p, 281, 1135)
    assert np.isnan(s) and np.isnan(r)


def test_read_cell_rejects_off_grid_and_missing_cell(tmp_path):
    p = str(tmp_path / "a.nc4")
    write_subset(p, 281, 1135, off_grid=True)
    with pytest.raises(RuntimeError, match="EASE-Grid"):
        sh.read_cell(p, 281, 1135)
    q = str(tmp_path / "b.nc4")
    write_subset(q, 281, 1135)
    with pytest.raises(RuntimeError, match="not in the returned subset"):
        sh.read_cell(q, 281, 1200)


def test_file_time_comes_from_the_file_name():
    t = sh._file_time("x/467799640_SMAP_L4_SM_gph_20261105T223000_Vv8013_001_subsetted.nc4")
    assert t == datetime(2026, 11, 5, 22, 30, tzinfo=UTC)


def test_second_request_is_served_from_cache(env):
    times = [T(1), T(8), T(15)]
    a = sh.fetch_times(times, LAT, LON, use_cache=True, **quiet)
    assert len(env.jobs) == 1 and len(env.jobs[0]) == 3
    b = sh.fetch_times(times, LAT, LON, use_cache=True, **quiet)
    assert len(env.jobs) == 1
    pd.testing.assert_frame_equal(a, b)


def test_only_missing_timestamps_are_fetched(env):
    sh.fetch_times([T(1), T(8)], LAT, LON, **quiet)
    sh.fetch_times([T(1), T(8), T(15), T(22)], LAT, LON, **quiet)
    assert env.jobs[1] == ["G-20261015T223000", "G-20261022T223000"]


def test_nearby_point_in_same_cell_reuses_the_cache(env):
    sh.fetch_times([T(1)], LAT, LON, **quiet)
    sh.fetch_times([T(1)], LAT + 0.001, LON + 0.001, **quiet)
    assert len(env.jobs) == 1


def test_missing_intervals_are_kept_as_nan_and_not_cached(env):
    env.no_granule = {T(8)}
    times = [T(15), T(1), T(8)]
    out = sh.fetch_times(times, LAT, LON, **quiet)
    assert list(out["time"]) == [T(1), T(8), T(15)]
    assert out["sm_surface"].isna().tolist() == [False, True, False]
    env.no_granule = set()
    sh.fetch_times(times, LAT, LON, **quiet)
    assert env.jobs[-1] == ["G-20261008T223000"]


def test_fill_value_cell_returns_nan_and_is_cached(env):
    env.fill_target = True
    out = sh.fetch_times([T(1)], LAT, LON, **quiet)
    assert out["sm_surface"].isna().all()
    sh.fetch_times([T(1)], LAT, LON, **quiet)
    assert len(env.jobs) == 1


def test_no_cache_flag_neither_reads_nor_writes(env):
    sh.fetch_times([T(1)], LAT, LON, use_cache=False, **quiet)
    sh.fetch_times([T(1)], LAT, LON, use_cache=False, **quiet)
    assert len(env.jobs) == 2
    assert not os.path.exists(sh.CELL_CACHE)


def test_timestamps_aligned_and_utc(env):
    out = sh.fetch_times([T(1), T(8)], LAT, LON, **quiet)
    assert all(t.tzinfo is not None and t.utcoffset() == timedelta(0) for t in out["time"])
    assert all(t.minute == 30 for t in out["time"])
    cached = pd.read_csv(sh.CELL_CACHE)
    assert cached["time"].tolist() == ["2026-10-01T22:30:00+00:00", "2026-10-08T22:30:00+00:00"]


def test_output_schema_matches_the_existing_pipeline(env):
    out = sh.fetch_times([T(1)], LAT, LON, **quiet)
    assert list(out.columns) == ["time", "sm_surface", "sm_rootzone"]
    import smap_timeseries
    assert smap_timeseries.fetch_times([T(1)], LAT, LON, backend="harmony")\
        .columns.tolist() == ["time", "sm_surface", "sm_rootzone"]


def test_cache_key_includes_product_version_and_cell(env):
    sh.fetch_times([T(1)], LAT, LON, **quiet)
    c = pd.read_csv(sh.CELL_CACHE, dtype=str)
    assert c.iloc[0][["product", "version", "row", "col"]].tolist() == ["SPL4SMGP", "008", "281", "1135"]
    m = pd.read_csv(sh.CELL_MAP)
    assert (m.iloc[0]["row"], m.iloc[0]["col"], bool(m.iloc[0]["verified"])) == (281, 1135, True)


def test_legacy_cache_is_imported_once(env, tmp_path):
    pd.DataFrame({"lat": [LAT], "lon": [LON], "time": [T(1).isoformat()],
                  "sm_surface": [0.21], "sm_rootzone": [0.26]}).to_csv(sh.LEGACY_CACHE, index=False)
    out = sh.fetch_times([T(1)], LAT, LON, **quiet)
    assert env.jobs == []
    assert out["sm_surface"].iloc[0] == pytest.approx(0.21)


def test_job_chunking_respects_the_limit(env, monkeypatch):
    monkeypatch.setattr(sh, "JOB_CHUNK", 2)
    sh.fetch_times([T(1), T(8), T(15), T(22), T(29)], LAT, LON, workers=1, **quiet)
    assert [len(j) for j in env.jobs] == [2, 2, 1]


def test_file_lock_is_exclusive(tmp_path):
    p = str(tmp_path / "x.lock")
    with sh._file_lock(p):
        with pytest.raises(TimeoutError):
            with sh._file_lock(p, timeout=0.6):
                pass
    with sh._file_lock(p, timeout=1):
        pass


def test_backend_must_be_explicit_no_silent_fallback():
    import smap_timeseries
    with pytest.raises(ValueError):
        smap_timeseries.fetch_times([T(1)], LAT, LON, backend="nonsense")


@pytest.mark.skipif(not os.environ.get("SMAP_LIVE_TEST"), reason="set SMAP_LIVE_TEST=1 for the live Harmony test")
def test_live_two_weeks(tmp_path, monkeypatch):
    monkeypatch.setattr(sh, "CELL_CACHE", str(tmp_path / "c.csv"))
    monkeypatch.setattr(sh, "CELL_MAP", str(tmp_path / "m.csv"))
    monkeypatch.setattr(sh, "LEGACY_CACHE", str(tmp_path / "none.csv"))
    end = datetime.now(UTC) - timedelta(days=10)
    times = [end.replace(hour=22, minute=30, second=0, microsecond=0) - timedelta(weeks=k) for k in (1, 0)]
    out = sh.fetch_times(times, LAT, LON)
    assert out["sm_surface"].between(0, 0.9).all() and out["sm_rootzone"].between(0, 0.9).all()
