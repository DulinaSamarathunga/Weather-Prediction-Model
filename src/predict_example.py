import os
import joblib
import pandas as pd

from preprocess import build_feature_frame  # note: no src. prefix

MODEL_PATH = os.path.join("models", "weather_model.joblib")


def main():
    # 1. Load trained pipeline
    pipeline = joblib.load(MODEL_PATH)

    # 2. Example input data
    input_data = {
        "Formatted Date": "2014-09-15 08:30:00.000 +0000",
        "Summary": "Clear",
        "Precip Type": None,          # or "snow" or None
        "Temperature (C)": 0,          # not used, just placeholder
        "Apparent Temperature (C)": 22.5,
        "Humidity": 0.45,
        "Wind Speed (km/h)": 12.0,
        "Wind Bearing (degrees)": 240.0,
        "Visibility (km)": 14.0,
        "Loud Cover": 0.0,
        "Pressure (millibars)": 1020.0,
        "Daily Summary": "Sunny and clear throughout the morning.",
    }

    df_input = pd.DataFrame([input_data])

    # 3. Build features
    X_new = build_feature_frame(df_input)

    # 4. Predict temperature
    predicted_temp = pipeline.predict(X_new)[0]
    print(f"Predicted Temperature (C): {predicted_temp:.2f}")


if __name__ == "__main__":
    main()
