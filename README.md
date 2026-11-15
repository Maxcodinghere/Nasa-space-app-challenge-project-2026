# SMAP soil-moisture forecast

A Streamlit app that forecasts surface and root-zone soil moisture for any location, from NASA SMAP L4
satellite data (9 km, one reading per week at the same time of day), with optional NASA POWER weather
features. An XGBoost model predicts 1 to 26 weeks ahead and reports how reliable each prediction was on
weeks it never saw.

## What you need

1. **Python 3.14.** Use 3.14 exactly: the forecasting model ships as `code/xgboost_model.pyc`, which is
   compiled for Python 3.14 and will not load on other versions. Check with `python --version`.
2. **A free NASA Earthdata account** (https://urs.earthdata.nasa.gov/users/new). It is used to download the
   SMAP data. See "Earthdata login" below.
3. An internet connection for the first run at a new location.

## Set up

Open a terminal in the project folder.

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS / Linux:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Earthdata login

Do ONE of these (the app never stores your password):

- **Type it in the app.** Open the "Earthdata login" box in the sidebar and enter your username and password.
- **Save it once** in a file named `_netrc` (Windows, in your user folder) or `.netrc` (macOS/Linux, in your
  home folder) with this content:

  ```
  machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD
  ```
- **Or set environment variables** `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`.

If a download is refused with an authorization error, sign in at https://urs.earthdata.nasa.gov, open
**Applications → Authorized Apps** and approve the NASA data applications it lists.

## Run the app

```powershell
python -m streamlit run code/app.py
```

Streamlit prints an address such as `http://localhost:8501`; open it in a browser. Pick a location in the
sidebar and press **Run forecast**. The first run for a new place downloads about 10 years of weekly data
(roughly a minute); later runs reuse the cache in `data/`. The weather features come from NASA POWER and need
no login.

To stop the app press `Ctrl+C` in the terminal. After you change any code, stop and start it again.

## Run the tests

```powershell
python -m pytest code/tests -q
```

The tests run offline. One live test is skipped unless you set `SMAP_LIVE_TEST=1`.

## What is in the project

| Path | What it is |
|---|---|
| `code/app.py` | The Streamlit app. |
| `code/smap_timeseries.py`, `code/smap_harmony.py` | Download SMAP data for one location (NASA Harmony, which downloads only a small subset of each file) and cache it per grid cell. |
| `code/weather_features.py` | NASA POWER weather features, added to the model as extra inputs and predicted as extra charts. |
| `code/xgboost_model.pyc` | The forecasting model (compiled; Python 3.14 only). |
| `code/hackathon_weekly.py` | Builds a 10-year weekly table from the data tools in `Weather data.py/` (NASA POWER, PRISM, NOAA NAO, ERA5), lined up with the soil-moisture weeks. |
| `code/original_xgboost.py` | The first version of the model, as a command-line script. |
| `Weather data.py/` | The weather and climate data tools from the NASA Hackathon work (a folder, despite the name). |
| `data/` | Cached downloads and the weekly datasets, so repeat runs are fast. |

## Optional extras

- **ERA5 climate data** in `hackathon_weekly.py` needs a Copernicus CDS token (set `CDSAPI_KEY`) and the
  `zarr` and `dask` packages. Without them, ERA5 columns are filled only where `Weather data.py/data/`
  already holds them (2025).
- **Old download method:** `python code/smap_timeseries.py LAT LON WEEKS --backend earthaccess` reads each
  global file instead of using Harmony. It is slower and only there for comparison.
