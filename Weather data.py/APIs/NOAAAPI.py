import requests
import pandas as pd


NAO_URL = (
    "https://www.cpc.ncep.noaa.gov/products/precip/"
    "CWlink/pna/norm.nao.monthly.b5001.current.ascii"
)


def fetch_nao_data(start_date, end_date):

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start > end:
        raise ValueError(
            "Start date must be before end date."
        )

    print("Downloading NOAA CPC NAO data...")

    response = requests.get(
        NAO_URL,
        timeout=60
    )

    response.raise_for_status()

    records = []

    for line in response.text.splitlines():

        values = line.split()

        if len(values) != 3:
            continue

        try:
            year = int(values[0])
            month = int(values[1])
            nao_value = float(values[2])

        except ValueError:
            continue

        if not 1 <= month <= 12:
            continue

        if abs(nao_value) >= 99:
            nao_value = float("nan")

        records.append({
            "Timestamp": pd.Timestamp(
                year=year,
                month=month,
                day=1
            ),
            "NAO_index": nao_value
        })

    if not records:
        raise ValueError(
            "Could not parse NOAA CPC NAO data."
        )

    df = pd.DataFrame(records)

    start_month = start.to_period("M")
    end_month = end.to_period("M")

    df = df[
        (df["Timestamp"].dt.to_period("M") >= start_month) &
        (df["Timestamp"].dt.to_period("M") <= end_month)
    ].copy()

    df = (
        df
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )

    if df["Timestamp"].duplicated().any():
        raise ValueError(
            "NOAA NAO returned duplicate monthly observations."
        )

    print(
        f"NOAA NAO completed: "
        f"{len(df)} monthly observations"
    )

    return df
