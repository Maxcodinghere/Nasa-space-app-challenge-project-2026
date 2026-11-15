from pathlib import Path

import pandas as pd

from APIs.NASApowerAPI import fetch_nasa_data
from APIs.PRISMAPI import fetch_prism_data
from APIs.ERA5API import fetch_era5_data
from APIs.NOAAAPI import fetch_nao_data


PARAMETERS = [
    "T2M",
    "T2MDEW",
    "T2MWET",
    "RH2M",
    "QV2M",
    "PS",
    "WS10M",
    "WD10M",
    "U10M",
    "V10M",
    "PRECTOT",
    "ALLSKY_SFC_SW_DWN"
]


PRISM_VARIABLES = [
    "tmin",
    "tmax",
    "tmean",
    "ppt"
]


VALID_INTERVALS = [1, 2, 3, 4, 6, 12, 24]


def normalize_timestamp(df):

    df = df.copy()

    df["Timestamp"] = (
        pd.to_datetime(
            df["Timestamp"],
            errors="coerce",
            utc=True
        )
        .dt.tz_convert(None)
    )

    return df


def generate_dataset(
    latitude,
    longitude,
    start_date,
    end_date,
    interval=1
):

    latitude = float(latitude)
    longitude = float(longitude)

    if not -90 <= latitude <= 90:
        raise ValueError("Invalid latitude.")

    if not -180 <= longitude <= 180:
        raise ValueError("Invalid longitude.")

    start_date = pd.Timestamp(start_date).date()
    end_date = pd.Timestamp(end_date).date()

    if start_date > end_date:
        raise ValueError("Invalid date range.")

    if interval not in VALID_INTERVALS:
        raise ValueError("Invalid sampling interval.")

    project_root = Path(__file__).resolve().parent

    data_folder = project_root / "data"
    data_folder.mkdir(parents=True, exist_ok=True)

    source_status = {}

    print("Retrieving NASA POWER data...")

    nasa_df = fetch_nasa_data(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        parameters=PARAMETERS,
        time_standard="UTC",
        community="AG"
    )

    if nasa_df.empty:
        raise ValueError(
            "NASA POWER returned no observations."
        )

    nasa_df = normalize_timestamp(nasa_df)

    nasa_df = nasa_df.dropna(
        subset=["Timestamp"]
    )

    if nasa_df["Timestamp"].duplicated().any():
        raise ValueError(
            "NASA POWER returned duplicate timestamps."
        )

    source_status["NASA POWER"] = (
        f"Retrieved {len(nasa_df)} hourly observations."
    )

    print("Retrieving hourly ERA5 data...")

    era5_folder = data_folder / "era5_cache"

    era5_df = fetch_era5_data(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        output_folder=era5_folder
    )

    if era5_df.empty:
        raise ValueError(
            "ERA5 returned no observations."
        )

    era5_df = normalize_timestamp(era5_df)

    era5_df = era5_df.dropna(
        subset=["Timestamp"]
    )

    if era5_df["Timestamp"].duplicated().any():
        raise ValueError(
            "ERA5 returned duplicate timestamps."
        )

    source_status["ERA5"] = (
        f"Retrieved {len(era5_df)} hourly observations."
    )

    print("Combining NASA POWER and ERA5...")

    combined_df = pd.merge(
        nasa_df,
        era5_df,
        on="Timestamp",
        how="outer",
        validate="one_to_one",
        sort=True
    )

    combined_df["Latitude"] = latitude
    combined_df["Longitude"] = longitude

    print("Retrieving PRISM data...")

    prism_cache = data_folder / "prism_cache"

    prism_columns = [
        f"PRISM_{variable}"
        for variable in PRISM_VARIABLES
    ]

    try:

        prism_df = fetch_prism_data(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            variables=PRISM_VARIABLES,
            cache_folder=prism_cache
        )

        if prism_df.empty:
            raise ValueError(
                "PRISM returned no observations."
            )

        prism_df = prism_df.copy()

        prism_df["Date"] = pd.to_datetime(
            prism_df["Date"],
            errors="coerce"
        ).dt.date

        prism_df = prism_df.dropna(
            subset=["Date"]
        )

        if prism_df["Date"].duplicated().any():
            raise ValueError(
                "PRISM returned duplicate daily observations."
            )

        for column in prism_columns:

            if column not in prism_df.columns:
                prism_df[column] = float("nan")

        combined_df["Date"] = (
            combined_df["Timestamp"].dt.date
        )

        combined_df = combined_df.merge(
            prism_df[["Date"] + prism_columns],
            on="Date",
            how="left",
            validate="many_to_one"
        )

        source_status["PRISM"] = (
            f"Retrieved {len(prism_df)} daily observations."
        )

    except Exception as error:

        for column in prism_columns:
            combined_df[column] = float("nan")

        source_status["PRISM"] = (
            f"Could not retrieve PRISM data: {error}"
        )

    combined_df.drop(
        columns=["Date"],
        inplace=True,
        errors="ignore"
    )

    print("Applying sampling interval...")

    combined_df = combined_df.dropna(
        subset=["Timestamp"]
    )

    if interval > 1:

        combined_df = combined_df[
            combined_df["Timestamp"].dt.hour % interval == 0
        ].copy()

    if combined_df.empty:
        raise ValueError(
            "No observations remain after filtering."
        )

    combined_df = (
        combined_df
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )

    print("Saving dataset without NAO...")

    output_path = (
        data_folder / "combined_dataset.csv"
    )

    combined_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Dataset without NAO saved: "
        f"{len(combined_df)} observations."
    )

    print("Retrieving monthly NOAA NAO data...")

    try:

        nao_df = fetch_nao_data(
            start_date=start_date,
            end_date=end_date
        )

        required_nao_columns = {
            "Timestamp",
            "NAO_index"
        }

        if not required_nao_columns.issubset(
            nao_df.columns
        ):
            raise ValueError(
                "NOAA NAO is missing required columns."
            )

        nao_df = nao_df.copy()

        nao_df["Timestamp"] = pd.to_datetime(
            nao_df["Timestamp"],
            errors="coerce"
        )

        nao_df["NAO_index"] = pd.to_numeric(
            nao_df["NAO_index"],
            errors="coerce"
        )

        nao_df = nao_df.dropna(
            subset=["Timestamp", "NAO_index"]
        ).copy()

        if nao_df["Timestamp"].duplicated().any():
            raise ValueError(
                "NOAA NAO returned duplicate monthly observations."
            )

        source_status["NOAA NAO"] = (
            f"Retrieved {len(nao_df)} valid monthly observations."
        )

    except Exception as error:

        print(
            f"Warning: Could not retrieve NOAA NAO data: {error}"
        )

        nao_df = pd.DataFrame(
            columns=["Timestamp", "NAO_index"]
        )

        source_status["NOAA NAO"] = (
            f"Could not retrieve NAO data: {error}"
        )

    print("Combining NOAA NAO with environmental data...")

    if nao_df.empty:

        df_with_nao = combined_df.iloc[0:0].copy()

        df_with_nao["NAO_index"] = pd.Series(
            dtype="float64"
        )

        source_status["NOAA NAO"] += (
            " No observations available for merging."
        )

    else:

        df_with_nao = combined_df.copy()

        df_with_nao["_NAO_month"] = (
            df_with_nao["Timestamp"].dt.to_period("M")
        )

        nao_df["_NAO_month"] = (
            nao_df["Timestamp"].dt.to_period("M")
        )

        df_with_nao = df_with_nao.merge(
            nao_df[["_NAO_month", "NAO_index"]],
            on="_NAO_month",
            how="inner",
            validate="many_to_one",
            sort=False
        )

        df_with_nao.drop(
            columns=["_NAO_month"],
            inplace=True
        )

        df_with_nao = (
            df_with_nao
            .sort_values("Timestamp")
            .reset_index(drop=True)
        )

        matched_months = (
            df_with_nao["Timestamp"]
            .dt.to_period("M")
            .nunique()
        )

        source_status["NOAA NAO"] = (
            f"Matched NAO to {len(df_with_nao)} "
            f"observations across {matched_months} months."
        )

    if df_with_nao["NAO_index"].isna().any():
        raise ValueError(
            "The dataset with NAO contains missing NAO values."
        )

    if df_with_nao.empty:
        print(
            "Warning: No observations with valid NAO values."
        )

    print("Saving dataset with NAO...")

    nao_output_path = (
        data_folder / "combined_dataset_with_nao.csv"
    )

    df_with_nao.to_csv(
        nao_output_path,
        index=False
    )

    print(
        f"Dataset with NAO saved: "
        f"{len(df_with_nao)} observations."
    )

    print("\n---------- DATASET GENERATION COMPLETE ----------")

    print(f"Without NAO: {output_path}")
    print(f"With NAO:    {nao_output_path}")

    print(
        f"\nObservations without NAO: {len(combined_df)}"
    )

    print(
        f"Observations with NAO: {len(df_with_nao)}"
    )

    print("\nSource status:")

    for source, status in source_status.items():

        print(f"{source}: {status}")

    combined_df.attrs["source_status"] = source_status

    return combined_df
