import contextlib
import os
import subprocess
import sys
import tempfile
from datetime import date, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smap_timeseries as sts  # noqa: E402  (cache location + valid slots)
import weather_features as wf  # noqa: E402  (NASA POWER weather from "Weather data.py"/APIs)
import xgboost_model as xm  # noqa: E402

import importlib  # noqa: E402
for _mod in (sts, wf):
    importlib.reload(_mod)

WEATHER_LABELS = {
    "rain_7d": "Rain, last 7 days (mm)",
    "rain_28d": "Rain, last 28 days (mm)",
    "t2m_7d": "Air temperature, 7-day mean (°C)",
    "rh2m_7d": "Relative humidity, 7-day mean (%)",
    "wind_7d": "Wind speed at 10 m, 7-day mean (m/s)",
}

st.set_page_config(page_title="Forecast predictions for the next weeks", layout="wide")
st.title("Forecast predictions for the next weeks")
st.caption("NASA SMAP L4 (9 km) · same weekday & 3-hour slot every week · XGBoost, 1–26 weeks ahead (you choose)")

with st.sidebar:
    st.header("Location & history")
    lat = st.number_input("Latitude", -90.0, 90.0, 48.4187, step=0.1, format="%.4f")
    lon = st.number_input("Longitude", -180.0, 180.0, -104.4876, step=0.1, format="%.4f")
    n_weeks = st.number_input(f"Weeks of history (min {xm.MIN_WEEKS})", xm.MIN_WEEKS, 520, 520, step=4,
                              help="More history = more reliable (520 weeks, about 10 years, gave the "
                                   "lowest validation error in testing). Under 52 weeks, time-of-year "
                                   "features are switched off. The first run for a new place downloads "
                                   "the weeks (about a minute); afterwards they come from the cache.")
    horizon = st.number_input("Weeks ahead to predict", 1, 26, 4, step=1,
                              help="How many weeks past the last observation to forecast. Further "
                                   "ahead is less certain: the ± (validation RMSE) is reported for "
                                   "each week, and training takes a little longer.")
    end_date = st.date_input("Most recent week", date.today() - timedelta(days=10),
                             max_value=date.today() - timedelta(days=4),
                             help="SMAP L4 data lags real time by ~3 days.")
    slot = st.selectbox("3-hour slot (UTC centre)", sts.VALID_SLOTS, index=len(sts.VALID_SLOTS) - 1)
    workers = st.slider("Parallel workers", 1, 16, 8)
    st.header("Additional features")
    weather_sel = st.multiselect(
        "Weather (NASA POWER)", list(WEATHER_LABELS), default=list(WEATHER_LABELS),
        format_func=WEATHER_LABELS.get,
        help="Extra model inputs from NASA POWER hourly weather at this location. Each is computed from "
             "the days BEFORE every weekly SMAP time, so nothing from the future leaks in. On the "
             "Montana test (520 weeks) they lowered validation error in 7 of 8 target/horizon cases "
             "(mean RMSE 0.0256 to 0.0252). Remove all to use soil moisture only. The first run for a "
             "place downloads the weather (cached afterwards).")
    center = st.checkbox(
        "Subtract each feature's mean before modelling", value=True,
        help="Every model input becomes (this week's value − that feature's mean over the whole history), "
             "so the model learns deviations from normal instead of absolute levels. The mean is added "
             "back to the predictions, so charts and tables stay in real units. For XGBoost this shifts "
             "values without changing which splits it makes, so the numbers come out the same; it is "
             "there for consistency and for any model that is sensitive to levels.")
    predict_weather = st.checkbox(
        "Also predict the weather features", value=True,
        help="Runs the same XGBoost model once more with each selected weather feature as the target, "
             "and charts its history and predicted weeks below the soil-moisture charts. Adds training "
             "time (roughly 10 s per feature at 4 weeks ahead, more for longer horizons). Temperature "
             "is predictable (it follows the seasons); rain mostly is not.")
    with st.expander("Earthdata login (only if not saved yet)"):
        st.caption("Leave empty if you already logged in once in a terminal. "
                   "Used only for this run; not stored.")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
    run = st.button("Run forecast", type="primary", use_container_width=True)
    st.caption(f"Cache: `{os.path.basename(sts.CACHE_PATH)}`")


class LiveLog:

    def __init__(self, placeholder):
        self.ph, self.text, self._partial = placeholder, "", ""

    def write(self, s):
        n = len(s)
        self._partial += s
        *lines, self._partial = self._partial.split("\n")
        s = "".join(ln + "\n" for ln in lines if "skill" not in ln.lower())
        if s:
            self.text += s
            self.ph.code(self.text[-6000:], language=None)
            sys.__stdout__.write(s)
        return n

    def flush(self):
        pass


def fetch(lat, lon, n_weeks, end, slot, workers, user, pwd, log):
    out_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    cmd = [sys.executable, "-u", os.path.join(HERE, "smap_timeseries.py"),
           str(lat), str(lon), str(n_weeks), "--end", str(end), "--slot", slot,
           "--workers", str(workers), "--csv", out_csv]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if user and pwd:
        env.update(EARTHDATA_USERNAME=user, EARTHDATA_PASSWORD=pwd)
    log.write("$ python smap_timeseries.py " + " ".join(cmd[3:-2]) + "\n")
    proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        log.write(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("Fetching failed (see the log above).")
    df = pd.read_csv(out_csv, parse_dates=["time"])
    os.remove(out_csv)
    return df


def show_table(df, decimals=4):
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d %H:%M")
    st.html('<div style="overflow-x:auto;max-height:420px;overflow-y:auto">'
            + out.to_html(index=False, na_rep="-", float_format=lambda v: f"{v:.{decimals}f}",
                          border=0, classes="smaptable") + "</div>"
            '<style>.smaptable{border-collapse:collapse;font-size:0.9rem}'
            '.smaptable th,.smaptable td{padding:4px 12px;text-align:right;'
            'border-bottom:1px solid #8884}</style>')


def _records(df):
    out = df.copy()
    out["time"] = out["time"].dt.tz_localize(None).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return out.astype(object).where(out.notna(), None).to_dict("records")


CHART_RANGES = {"Last 12 months + predictions": 365, "Last 6 months + predictions": 183,
                "Last 3 years + predictions": 1095, "All history": None}


def make_chart(hist, fc, col, title, unit="m³/m³", bounds=(None, None), window_days=None, decimals=3,
               normalize=False):
    h = hist[["time", col]].dropna().rename(columns={col: "value"})
    f = fc[["time", col, col + "_rmse"]].rename(columns={col: "value", col + "_rmse": "err"})
    f["lo"], f["hi"] = f["value"] - f["err"], f["value"] + f["err"]
    f["lo"], f["hi"] = f["lo"].clip(lower=bounds[0], upper=bounds[1]), f["hi"].clip(lower=bounds[0], upper=bounds[1])
    if normalize:
        lo_v, hi_v = float(h["value"].min()), float(h["value"].max())
        rng = (hi_v - lo_v) or 1.0
        h["value"] = (h["value"] - lo_v) / rng
        for c in ("value", "lo", "hi"):
            f[c] = (f[c] - lo_v) / rng
        f["err"] = f["err"] / rng
        unit = "0–1 (per feature)"
    last = h.tail(1).assign(lo=h["value"].iloc[-1], hi=h["value"].iloc[-1], err=0.0)
    link = pd.concat([last, f], ignore_index=True)
    if window_days:
        h = h[h["time"] >= h["time"].iloc[-1] - pd.Timedelta(days=window_days)]
    fmt = f".{decimals}f"
    week = alt.Tooltip("time:T", title="Week", format="%Y-%m-%d")
    x = alt.X("time:T", title=None)
    y = alt.Y("value:Q", title=unit, scale=alt.Scale(zero=False))
    obs = alt.Chart(alt.Data(values=_records(h))).mark_line(point=True, color="#1f77b4").encode(
        x, y, tooltip=[week, alt.Tooltip("value:Q", title=f"Observed ({unit})", format=fmt)])
    pred = alt.Chart(alt.Data(values=_records(link))).mark_line(
        point=True, color="#d62728", strokeDash=[5, 3]).encode(
        x, y, tooltip=[week, alt.Tooltip("value:Q", title=f"Predicted ({unit})", format=fmt),
                       alt.Tooltip("lo:Q", title="Low (− error)", format=fmt),
                       alt.Tooltip("hi:Q", title="High (+ error)", format=fmt)])
    band = alt.Chart(alt.Data(values=_records(link))).mark_area(opacity=0.2, color="#d62728").encode(
        x, alt.Y("lo:Q"), alt.Y2("hi:Q"))
    return (band + obs + pred).properties(title=title, height=300).interactive()


SOIL_NAMES = {"sm_surface": "Surface soil moisture (m³/m³)", "sm_rootzone": "Root-zone soil moisture (m³/m³)"}


def reliability_table(metrics, forecast, names, hist):
    m = metrics.reset_index(drop=True)
    pred = [float(forecast[t].iloc[int(w) - 1]) for t, w in zip(m["target"], m["weeks_ahead"])]
    spread = {t: (float(hist[t].max() - hist[t].min()), float(hist[t].std())) for t in m["target"].unique()}
    pct_range = [f"{100 * e / spread[t][0]:.1f} %" if spread[t][0] else "n/a" for t, e in zip(m["target"], m["mae"])]
    in_std = [e / spread[t][1] if spread[t][1] else np.nan for t, e in zip(m["target"], m["mae"])]
    return pd.DataFrame({
        "What is predicted": m["target"].map(names),
        "Weeks ahead": m["weeks_ahead"],
        "Predicted value": pred,
        "Average error (MAE)": m["mae"],
        "Error as % of the feature's range": pct_range,
        "Error in std units (MAE ÷ std)": in_std,
        "Typical error (RMSE)": m["rmse"],
        "Correlation (1 = perfect)": m["correlation"],
        "Test weeks used": m["scored_rows"],
    })


if run:
    st.session_state.pop("result", None)
    try:
        do_wfc = bool(weather_sel) and predict_weather
        n_steps = 2 + bool(weather_sel) + do_wfc
        step = 1
        with st.status(f"{step}/{n_steps}  Fetching SMAP data (cached weeks are reused) ...", expanded=True) as s1:
            log1 = LiveLog(st.empty())
            hist = fetch(lat, lon, int(n_weeks), end_date, slot, workers, user, pwd, log1)
            s1.update(label=f"{step}/{n_steps}  Data ready: {hist['sm_surface'].notna().sum()} of {len(hist)} weeks have values",
                      state="complete", expanded=False)
        features = None
        if weather_sel:
            step += 1
            with st.status(f"{step}/{n_steps}  Adding NASA POWER weather features ...", expanded=True) as sw:
                logw = LiveLog(st.empty())
                with contextlib.redirect_stdout(logw):
                    hist = wf.add_weather(hist, lat, lon)
                full = int(hist[weather_sel].notna().all(axis=1).sum())
                sw.update(label=f"{step}/{n_steps}  Weather ready: {full} of {len(hist)} weeks complete",
                          state="complete", expanded=False)
            features = list(xm.TARGETS) + list(weather_sel)
        input_cols = list(xm.TARGETS) + list(weather_sel)
        means = hist[input_cols].mean() if center else None
        model_df = hist.copy()
        if center:
            model_df[input_cols] = model_df[input_cols] - means
        step += 1
        with st.status(f"{step}/{n_steps}  Training XGBoost and forecasting soil moisture ...", expanded=True) as s2:
            log2 = LiveLog(st.empty())
            with contextlib.redirect_stdout(log2):
                res = xm.forecast(model_df, horizon=int(horizon), features=features)
            if center:
                res = dict(res, forecast=res["forecast"].assign(
                    **{c: res["forecast"][c] + means[c] for c in xm.TARGETS}))
            s2.update(label=f"{step}/{n_steps}  Soil-moisture forecast done", state="complete", expanded=False)
        wres = None
        if do_wfc:
            step += 1
            with st.status(f"{step}/{n_steps}  Predicting the weather features (one more ML pass) ...", expanded=True) as s3:
                log3 = LiveLog(st.empty())
                with contextlib.redirect_stdout(log3):
                    wres = wf.forecast_weather(model_df, int(horizon), weather_sel, features,
                                               offsets=means.to_dict() if center else None)
                s3.update(label=f"{step}/{n_steps}  Weather forecast done", state="complete", expanded=False)
        st.session_state["result"] = dict(hist=hist, loc=(lat, lon), n=int(n_weeks), centered=bool(center),
                                          weather=list(weather_sel), weather_res=wres, **res)
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")

res = st.session_state.get("result")
if res is None:
    st.info("Choose a location and press **Run forecast**. The first run for a place downloads "
            "the weeks; afterwards they come from the cache.")
else:
    hist, fc, metrics = res["hist"], res["forecast"], res["metrics"]
    st.subheader(f"Forecast for {res['loc'][0]:.3f}, {res['loc'][1]:.3f}  ·  {res['n']} weeks of history")

    soil = hist[["sm_surface", "sm_rootzone"]].dropna()
    last = soil.iloc[-1] if len(soil) else None
    used = res.get("weather") or []
    st.caption("Model inputs: soil moisture (surface + root zone)" +
               (" + weather: " + ", ".join(WEATHER_LABELS[k] for k in used) if used else " only") +
               ("  ·  each input is (value − its mean) before modelling" if res.get("centered") else ""))
    view = st.selectbox("Chart range", list(CHART_RANGES), index=0,
                        help="Charts are interactive: scroll to zoom, drag to pan, double-click to reset, "
                             "hover a point for its week and value. Pick a range here to jump straight to "
                             "the recent weeks and the predictions.")
    days = CHART_RANGES[view]
    norm = st.checkbox("Normalize charts per feature (0–1)", value=False,
                       help="Rescales every chart to its own 0–1 range (0 = that feature's lowest week in "
                            "the history, 1 = its highest), so features with different units can be compared "
                            "side by side. Hover values then show the normalized number.")
    c1, c2 = st.columns(2)
    for c, col, name in ((c1, "sm_surface", "Surface soil moisture"), (c2, "sm_rootzone", "Root-zone soil moisture")):
        nxt = fc[col].iloc[0]
        c.metric(f"{name} · next week", f"{nxt:.3f} m³/m³",
                 None if last is None else f"{nxt - last[col]:+.3f} vs last observed")
        c.altair_chart(make_chart(hist, fc, col, name, window_days=days, decimals=4, normalize=norm),
                       use_container_width=True)

    wres = res.get("weather_res")
    if wres is not None:
        wfc = wres["forecast"]
        st.markdown("##### Weather features: history and predicted weeks")
        st.caption("Blue = observed (NASA POWER), red dashed = predicted by XGBoost, shaded = ± validation RMSE. "
                   "Each value is the 7- or 28-day aggregate ending at that week's SMAP time.")
        cols = st.columns(2)
        for i, k in enumerate(used):
            cols[i % 2].altair_chart(
                make_chart(hist, wfc, k, WEATHER_LABELS[k], unit=wf.WEATHER_UNITS[k],
                           bounds=wf.WEATHER_BOUNDS[k], window_days=days, decimals=2, normalize=norm),
                use_container_width=True)

    st.markdown(f"##### {len(fc)}-week forecast")
    show_table(fc.rename(columns={"sm_surface_rmse": "± surface", "sm_rootzone_rmse": "± rootzone"}))
    st.download_button("Download forecast CSV", fc.to_csv(index=False), "forecast.csv", "text/csv")
    if wres is not None:
        st.markdown("##### Predicted weather features")
        show_table(wfc.rename(columns={k: WEATHER_LABELS[k] for k in used} |
                                      {k + "_rmse": "± " + WEATHER_LABELS[k].split(" (")[0] for k in used}), 3)
        st.download_button("Download weather forecast CSV", wfc.to_csv(index=False), "weather_forecast.csv", "text/csv")

    with st.expander("How reliable? (sliding-window validation scores)"):
        st.markdown("**Table 1: Soil moisture**")
        show_table(reliability_table(metrics, fc, SOIL_NAMES, hist), 4)
        if wres is not None:
            st.markdown("**Table 2: Weather features**")
            wnames = {k: WEATHER_LABELS[k] for k in used}
            show_table(reliability_table(wres["metrics"], wfc, wnames, hist), 2)
        st.caption("Each row was scored on weeks the model never saw while training (sliding-window "
                   "validation). **Average error (MAE)**: how far off the prediction was, on average, in the "
                   "value's own units. The next two columns normalize that error PER FEATURE using the "
                   "feature's own history, so different features are comparable: **Error as % of the "
                   "feature's range** = MAE ÷ (highest week − lowest week), and **Error in std units** = "
                   "MAE ÷ the feature's standard deviation (about 0.8 means the miss is nearly as big as the "
                   "feature's own typical week-to-week swing; smaller is better). **Typical error (RMSE)**: "
                   "like MAE but big misses count more; it is the ± shown in the forecast. **Correlation**: 1 means the "
                   "predictions rise and fall exactly with the real values, 0 means no relationship. "
                   "**Test weeks used**: how many unseen weeks were scored.")
    with st.expander("Most important features"):
        show_table(res["importance"].reset_index(names="feature"), 3)
    with st.expander("Weekly data used (the table behind the model)"):
        show_table(hist)
        st.download_button("Download data CSV", hist.to_csv(index=False), "weekly_data.csv", "text/csv")
