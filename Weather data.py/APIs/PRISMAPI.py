import requests
import pandas as pd


BASE_URL = "https://data.rcc-acis.org/GridData"

GRID = "21"

VARIABLES = {
    "tmin": {
        "api_name": "mint",
        "units": "degreeC"
    },
    "tmax": {
        "api_name": "maxt",
        "units": "degreeC"
    },
    "tmean": {
        "api_name": "avgt",
        "units": "degreeC"
    },
    "ppt": {
        "api_name": "pcpn",
        "units": "mm"
    }
}


def convert_value(value):

    if value is None or value == "M":
        return float("nan")

    if value == "T":
        return 0.0

    if isinstance(value, str) and value.endswith("A"):
        value = value[:-1]

    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def fetch_prism_data(
    latitude,
    longitude,
    start_date,
    end_date,
    variables,
    cache_folder=None
):

    if not -90 <= latitude <= 90:
        raise ValueError("Invalid latitude.")

    if not -180 <= longitude <= 180:
        raise ValueError("Invalid longitude.")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if pd.isna(start) or pd.isna(end):
        raise ValueError("Invalid date.")

    if start > end:
        raise ValueError("Start date must precede end date.")

    if start < pd.Timestamp("1981-01-01"):
        raise ValueError(
            "PRISM daily data starts on January 1, 1981."
        )

    if not variables:
        raise ValueError(
            "Select at least one PRISM variable."
        )

    invalid_variables = [
        variable for variable in variables
        if variable not in VARIABLES
    ]

    if invalid_variables:
        raise ValueError(
            f"Unsupported PRISM variables: {invalid_variables}. "
            f"Available variables: {list(VARIABLES)}"
        )

    elements = []

    for variable in variables:

        configuration = VARIABLES[variable]

        elements.append({
            "name": configuration["api_name"],
            "interval": "dly",
            "units": configuration["units"]
        })

    params = {
        "loc": f"{longitude},{latitude}",
        "sdate": start.strftime("%Y-%m-%d"),
        "edate": end.strftime("%Y-%m-%d"),
        "grid": GRID,
        "elems": elements,
        "meta": ["ll", "elev"]
    }

    try:
        response = requests.post(
            BASE_URL,
            json=params,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            timeout=180
        )

        response.raise_for_status()

        result = response.json()

    except requests.RequestException as error:
        raise ConnectionError(
            f"Could not retrieve PRISM data: {error}"
        ) from error

    except ValueError as error:
        raise ValueError(
            "PRISM returned an invalid JSON response."
        ) from error

    if "error" in result:
        raise ValueError(
            f"PRISM API error: {result['error']}"
        )

    if "data" not in result:
        raise ValueError(
            "PRISM response does not contain climate data."
        )

    rows = result["data"]

    if not rows:
        raise ValueError(
            "PRISM returned no observations."
        )

    column_names = [
        "Date"
    ] + [
        f"PRISM_{variable}"
        for variable in variables
    ]

    records = []

    for row in rows:

        if len(row) != len(column_names):
            raise ValueError(
                f"Unexpected PRISM data format: {row}"
            )

        record = {
            "Date": pd.to_datetime(row[0]).date()
        }

        for index, variable in enumerate(variables):

            record[f"PRISM_{variable}"] = convert_value(
                row[index + 1]
            )

        records.append(record)

    df = pd.DataFrame(records)

    df = df.sort_values("Date").reset_index(drop=True)

    return df
