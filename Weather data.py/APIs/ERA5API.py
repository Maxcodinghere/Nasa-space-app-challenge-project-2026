import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ZARR_URL = (
    "https://arco.datastores.ecmwf.int/"
    "cadl-arco-geo-002/arco/"
    "reanalysis_era5_single_levels/sfc/geoChunked.zarr"
)

VARIABLE_NAMES = {
    "t2m": "ERA5_temperature_2m_C",
    "d2m": "ERA5_dewpoint_2m_C",
    "u10": "ERA5_u_wind_10m_m_s",
    "v10": "ERA5_v_wind_10m_m_s",
    "sp": "ERA5_surface_pressure_kPa",
    "tp": "ERA5_total_precipitation_mm",
    "ssrd": "ERA5_solar_radiation_J_m2"
}

ARCO_VARIABLES = list(VARIABLE_NAMES.keys())


def get_cds_api_key():

    api_key = os.environ.get("CDSAPI_KEY")

    if api_key:
        return api_key.strip()

    project_root = Path(__file__).resolve().parents[1]

    possible_files = []

    cdsapi_rc = os.environ.get("CDSAPI_RC")

    if cdsapi_rc:
        possible_files.append(Path(cdsapi_rc))

    possible_files.extend([
        project_root / ".cdsapirc",
        Path.home() / ".cdsapirc"
    ])

    for config_file in possible_files:

        if not config_file.is_file():
            continue

        with config_file.open(
            "r",
            encoding="utf-8-sig"
        ) as file:

            for line in file:

                line = line.strip()

                if line.startswith("key:"):

                    api_key = line.split(":", 1)[1].strip()

                    api_key = api_key.strip("\"'")

                    if api_key:
                        return api_key

    raise RuntimeError(
        "CDS API key not found. Set CDSAPI_KEY or "
        "configure .cdsapirc in your project folder."
    )


def get_cache_path(
    output_folder,
    latitude,
    longitude,
    start,
    end
):

    filename = (
        f"era5_arco_"
        f"lat{latitude:.5f}_"
        f"lon{longitude:.5f}_"
        f"{start:%Y%m%d}_"
        f"{end:%Y%m%d}.csv"
    )

    return output_folder / filename


def fetch_era5_data(
    latitude,
    longitude,
    start_date,
    end_date,
    output_folder
):

    latitude = float(latitude)
    longitude = float(longitude)

    if not -90 <= latitude <= 90:
        raise ValueError("Invalid latitude.")

    if not -180 <= longitude <= 180:
        raise ValueError("Invalid longitude.")

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    if start > end:
        raise ValueError("Invalid date range.")

    last_hour = (
        end + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    )

    output_folder = Path(output_folder)

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    cache_path = get_cache_path(
        output_folder,
        latitude,
        longitude,
        start,
        end
    )

    csv_path = output_folder / "era5_dataset.csv"

    required_columns = {
        "Timestamp",
        *VARIABLE_NAMES.values(),
        "ERA5_wind_speed_10m_m_s",
        "ERA5_wind_direction_10m_deg",
        "ERA5_solar_radiation_W_m2"
    }

    if cache_path.exists():

        try:

            cached_df = pd.read_csv(
                cache_path,
                parse_dates=["Timestamp"]
            )

            if (
                not cached_df.empty
                and required_columns.issubset(cached_df.columns)
            ):

                cached_df["Timestamp"] = pd.to_datetime(
                    cached_df["Timestamp"]
                )

                cached_df = cached_df[
                    (cached_df["Timestamp"] >= start) &
                    (cached_df["Timestamp"] <= last_hour)
                ].copy()

                if not cached_df.empty:

                    cached_df = (
                        cached_df
                        .sort_values("Timestamp")
                        .reset_index(drop=True)
                    )

                    cached_df.to_csv(
                        csv_path,
                        index=False
                    )

                    print(
                        f"[ERA5 CACHE] Using local data: "
                        f"{len(cached_df)} observations"
                    )

                    return cached_df

        except (OSError, ValueError, KeyError, pd.errors.ParserError):

            print(
                "[ERA5 CACHE] Existing cache is invalid. "
                "Retrieving data again."
            )

    print("Connecting to ERA5 ARCO Zarr...")

    api_key = get_cds_api_key()

    storage_options = {
        "headers": {
            "Authorization": f"Bearer {api_key}"
        }
    }

    print("Opening ERA5 remote dataset...")

    try:

        ds = xr.open_zarr(
            ZARR_URL,
            consolidated=True,
            storage_options=storage_options
        )

    except Exception as error:

        raise RuntimeError(
            "Could not access the ERA5 ARCO Zarr store. "
            "Check your CDS API token, dataset licence "
            "acceptance, internet connection, and Zarr "
            f"dependencies. Original error: {error}"
        ) from error

    try:

        missing_variables = [
            variable
            for variable in ARCO_VARIABLES
            if variable not in ds.data_vars
        ]

        if missing_variables:

            raise ValueError(
                "ERA5 ARCO is missing expected variables: "
                f"{missing_variables}. Available variables: "
                f"{list(ds.data_vars)}"
            )

        if "latitude" not in ds.coords:
            raise ValueError(
                "ERA5 dataset has no latitude coordinate."
            )

        if "longitude" not in ds.coords:
            raise ValueError(
                "ERA5 dataset has no longitude coordinate."
            )

        if "time" in ds.coords:
            time_column = "time"

        elif "valid_time" in ds.coords:
            time_column = "valid_time"

        else:
            raise ValueError(
                "ERA5 dataset has no recognized time coordinate."
            )

        selected = ds[ARCO_VARIABLES]

        longitude_min = float(ds["longitude"].min())
        longitude_max = float(ds["longitude"].max())

        selected_longitude = longitude

        if longitude_min >= 0 and longitude < 0:
            selected_longitude = longitude % 360

        elif longitude_max > 180 and longitude < 0:
            selected_longitude = longitude % 360

        point = selected.sel(
            latitude=latitude,
            longitude=selected_longitude,
            method="nearest"
        )

        grid_latitude = float(
            point["latitude"].values
        )

        grid_longitude = float(
            point["longitude"].values
        )

        print(
            f"Selected ERA5 grid point: "
            f"{grid_latitude}, {grid_longitude}"
        )

        point = point.sel({
            time_column: slice(
                start.to_datetime64(),
                last_hour.to_datetime64()
            )
        })

        df = point.to_dataframe().reset_index()

    finally:

        ds.close()

    if time_column not in df.columns:

        raise ValueError(
            "Could not find timestamp in ERA5 results."
        )

    df.rename(
        columns={
            time_column: "Timestamp"
        },
        inplace=True
    )

    df["Timestamp"] = (
        pd.to_datetime(
            df["Timestamp"],
            errors="coerce",
            utc=True
        )
        .dt.tz_convert(None)
    )

    df = df.dropna(
        subset=["Timestamp"]
    ).copy()

    df = df[
        (df["Timestamp"] >= start) &
        (df["Timestamp"] <= last_hour)
    ].copy()

    df = (
        df
        .drop_duplicates(subset=["Timestamp"])
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "ERA5 returned no hourly observations "
            "for the requested location and dates."
        )

    df.rename(
        columns=VARIABLE_NAMES,
        inplace=True
    )

    for column in [
        "ERA5_temperature_2m_C",
        "ERA5_dewpoint_2m_C"
    ]:

        df[column] = df[column] - 273.15

    df["ERA5_surface_pressure_kPa"] /= 1000

    df["ERA5_total_precipitation_mm"] *= 1000

    u = df["ERA5_u_wind_10m_m_s"]
    v = df["ERA5_v_wind_10m_m_s"]

    df["ERA5_wind_speed_10m_m_s"] = np.sqrt(
        u ** 2 + v ** 2
    )

    df["ERA5_wind_direction_10m_deg"] = (
        180 + np.degrees(
            np.arctan2(u, v)
        )
    ) % 360

    df["ERA5_solar_radiation_W_m2"] = (
        df["ERA5_solar_radiation_J_m2"] / 3600
    )

    df.to_csv(
        cache_path,
        index=False
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print(
        f"ERA5 completed: {len(df)} hourly observations"
    )

    print(
        f"ERA5 cache saved to: {cache_path}"
    )

    print(
        f"ERA5 dataset saved to: {csv_path}"
    )

    return df
