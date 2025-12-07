import os
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime, date, time

from preprocess import build_feature_frame  # uses your existing preprocessing


# Path to the saved model (models/weather_model.joblib)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "weather_model.joblib")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def main():
    st.title("🌤 Weather Temperature Prediction")
    st.write("Enter weather conditions to predict the **temperature (°C)** using your Linear Regression model.")

    model = load_model()

    st.subheader("📅 Date & Time")

    col1, col2 = st.columns(2)
    with col1:
        input_date = st.date_input("Date", value=date(2014, 9, 15))
    with col2:
        input_time = st.time_input("Time", value=time(8, 30))

    # Build Formatted Date string like in the dataset
    formatted_date = datetime.combine(input_date, input_time).strftime("%Y-%m-%d %H:%M:%S.000 +0000")

    st.subheader("🌡️ Weather Inputs")

    col_a, col_b = st.columns(2)

    with col_a:
        apparent_temp = st.number_input("Apparent Temperature (°C)", value=22.5)
        humidity = st.slider("Humidity (0–1)", min_value=0.0, max_value=1.0, value=0.45, step=0.01)
        wind_speed = st.number_input("Wind Speed (km/h)", value=12.0, min_value=0.0)
    with col_b:
        wind_bearing = st.number_input("Wind Bearing (degrees)", value=240.0, min_value=0.0, max_value=360.0)
        visibility = st.number_input("Visibility (km)", value=14.0, min_value=0.0)
        pressure = st.number_input("Pressure (millibars)", value=1020.0, min_value=800.0, max_value=1100.0)

    st.subheader("🌧 Precipitation & Text (Optional)")

    precip_type = st.selectbox("Precipitation Type", ["none", "rain", "snow"])
    summary = st.text_input("Summary (optional)", value="Clear")
    daily_summary = st.text_area("Daily Summary (optional)", value="Sunny and clear throughout the day.")

    # Build input dictionary exactly like your CSV columns
    input_data = {
        "Formatted Date": formatted_date,
        "Summary": summary,
        "Precip Type": precip_type if precip_type != "none" else None,
        "Temperature (C)": 0.0,  # placeholder, not used
        "Apparent Temperature (C)": apparent_temp,
        "Humidity": humidity,
        "Wind Speed (km/h)": wind_speed,
        "Wind Bearing (degrees)": wind_bearing,
        "Visibility (km)": visibility,
        "Loud Cover": 0.0,  # dataset uses this; keep at 0
        "Pressure (millibars)": pressure,
        "Daily Summary": daily_summary,
    }

    if st.button("🔮 Predict Temperature"):
        df_input = pd.DataFrame([input_data])

        # Use same preprocessing as training
        X_new = build_feature_frame(df_input)

        predicted_temp = model.predict(X_new)[0]

        st.success(f"Predicted Temperature: **{predicted_temp:.2f} °C**")


if __name__ == "__main__":
    main()