import requests
import pandas as pd


BASE_URL = (
    "https://power.larc.nasa.gov/api/temporal/hourly/point"
)


def fetch_nasa_data(
    latitude,
    longitude,
    start_date,
    end_date,
    parameters,
    time_standard="LST",
    community="AG"
):

    params = {
        "parameters": ",".join(parameters),
        "community": community,
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON",
        "time-standard": time_standard
    }

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    messages = result.get("messages", [])

    if messages:
        errors = [
            str(message)
            for message in messages
            if "error" in str(message).lower()
        ]

        if errors:
            raise ValueError("\n".join(errors))

    try:
        parameter_data = result["properties"]["parameter"]
    except (KeyError, TypeError):
        raise ValueError(
            "NASA returned an unexpected response. "
            "Check the coordinates, dates, and parameters."
        )

    data_series = []

    for parameter, observations in parameter_data.items():

        timestamps = pd.to_datetime(
            list(observations.keys()),
            format="%Y%m%d%H",
            errors="coerce"
        )

        values = pd.to_numeric(
            list(observations.values()),
            errors="coerce"
        )

        series = pd.Series(
            values,
            index=timestamps,
            name=parameter
        )

        data_series.append(series)

    if not data_series:
        raise ValueError("NASA returned no weather observations.")

    df = pd.concat(data_series, axis=1)

    df.index.name = "Timestamp"

    df = df.sort_index()

    df = df.mask(df == -999)

    df.insert(0, "Longitude", longitude)
    df.insert(0, "Latitude", latitude)

    df = df.reset_index()

    return df
