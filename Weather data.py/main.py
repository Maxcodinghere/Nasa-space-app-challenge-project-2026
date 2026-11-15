import streamlit as st
from datetime import date

from pipeline import generate_dataset


st.set_page_config(
    page_title="Environmental Dataset Generator",
    page_icon="🌱",
    layout="wide"
)


st.title("🌱 Environmental Dataset Generator")

st.write(
    """
    Generate machine-learning-ready environmental datasets
    using NASA POWER, PRISM, and ERA5 historical climate data.

    Enter a location, select a date range, and generate
    a combined CSV dataset for further analysis and model training.
    """
)

st.divider()


st.subheader("1. Location")

col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=48.424274,
        format="%.6f"
    )

with col2:
    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=-104.486416,
        format="%.6f"
    )


st.divider()
st.subheader("2. Date range")

col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input(
        "Start date",
        value=date(2025, 1, 1)
    )

with col2:
    end_date = st.date_input(
        "End date",
        value=date(2025, 12, 31)
    )


st.divider()
st.subheader("3. Sampling interval")

interval = st.selectbox(
    "Retrieve one observation every:",
    options=[1, 2, 3, 4, 6, 12, 24],
    index=0,
    format_func=lambda x: f"{x} hour(s)"
)

st.caption(
    "For example, selecting 1 hour keeps every hourly "
    "observation, while 6 hours keeps one observation "
    "every six hours."
)


st.divider()
st.subheader("4. Dataset information")

st.write(
    """
    The application automatically combines:

    - NASA POWER: hourly meteorological observations.
    - ERA5: hourly global climate reanalysis.
    - PRISM: daily temperature and precipitation data
      where geographic coverage is available.

    The resulting CSV includes temperature, humidity,
    atmospheric pressure, wind, precipitation, and
    solar radiation variables.
    """
)

st.info(
    "All three sources are attempted automatically. "
    "PRISM does not cover every location, including Canada. "
    "If unavailable, its columns will contain missing values."
)

st.caption(
    "Hourly timestamps are stored in UTC. "
    "ERA5 uses a nearest grid point rather than an exact "
    "measurement at the entered coordinates."
)


st.divider()

if st.button(
    "Generate CSV dataset",
    type="primary",
    use_container_width=True
):

    if start_date > end_date:

        st.error(
            "The start date must be before the end date."
        )

    else:

        try:

            with st.spinner(
                "Retrieving NASA POWER, ERA5, and PRISM data. "
                "This may take several minutes..."
            ):

                df = generate_dataset(
                    latitude=latitude,
                    longitude=longitude,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval
                )

            st.success(
                "Combined dataset successfully generated!"
            )

            st.subheader("Data source status")

            source_status = df.attrs.get(
                "source_status",
                {}
            )

            for source, status in source_status.items():

                if "Unavailable" in status:
                    st.warning(f"**{source}:** {status}")
                else:
                    st.success(f"**{source}:** {status}")

            st.subheader("Dataset statistics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Observations",
                    len(df)
                )

            with col2:
                st.metric(
                    "Variables",
                    len(df.columns) - 3
                )

            with col3:
                st.metric(
                    "Sampling interval",
                    f"{interval} hour(s)"
                )

            st.subheader("Dataset preview")

            st.dataframe(
                df.head(20),
                use_container_width=True
            )

            st.subheader("Missing data")

            missing_data = df.isna().sum()

            st.dataframe(
                missing_data.rename("Missing values"),
                use_container_width=True
            )

            csv_data = df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download combined CSV",
                data=csv_data,
                file_name="combined_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.success(
                "A copy of the combined dataset has also "
                "been saved in the project's data folder."
            )

        except Exception as error:

            st.error(
                f"Dataset generation failed: {error}"
            )
