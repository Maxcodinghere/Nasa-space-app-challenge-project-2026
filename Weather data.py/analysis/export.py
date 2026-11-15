import pandas as pd
from io import BytesIO


def create_csv(df):

    return df.to_csv(index=False).encode("utf-8")


def create_excel(df, metadata):

    excel_buffer = BytesIO()

    weather_columns = [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in ["Latitude", "Longitude"]
    ]

    weather_data = df[weather_columns]

    statistics = (
        weather_data
        .agg(["count", "mean", "std", "min", "max"])
        .T
        .reset_index()
    )

    statistics.columns = [
        "Parameter",
        "Count",
        "Mean",
        "Standard Deviation",
        "Minimum",
        "Maximum"
    ]

    missing_data = pd.DataFrame({
        "Parameter": weather_columns,
        "Missing Count": weather_data.isna().sum().values,
        "Missing Percentage": (
            weather_data.isna().mean().values * 100
        ).round(2)
    })

    metadata_df = pd.DataFrame(
        list(metadata.items()),
        columns=["Property", "Value"]
    )

    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="Weather Data",
            index=False
        )

        statistics.to_excel(
            writer,
            sheet_name="Statistics",
            index=False
        )

        missing_data.to_excel(
            writer,
            sheet_name="Missing Data",
            index=False
        )

        metadata_df.to_excel(
            writer,
            sheet_name="Metadata",
            index=False
        )

    return excel_buffer.getvalue()
