import pandas as pd

# Numeric feature columns
NUM_COLS = [
    "Apparent Temperature (C)",
    "Humidity",
    "Wind Speed (km/h)",
    "Wind Bearing (degrees)",
    "Visibility (km)",
    "Loud Cover",
    "Pressure (millibars)",
    "year",
    "month",
    "day",
    "hour",
    "dayofweek",
]

# Categorical feature columns
CAT_COLS = [
    "Precip Type",
]

FEATURE_COLS = NUM_COLS + CAT_COLS


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add year, month, day, hour, dayofweek from Formatted Date."""
    df = df.copy()

    df["Formatted Date"] = pd.to_datetime(
        df["Formatted Date"],
        errors="coerce",
        utc=True
    )

    df["year"] = df["Formatted Date"].dt.year
    df["month"] = df["Formatted Date"].dt.month
    df["day"] = df["Formatted Date"].dt.day
    df["hour"] = df["Formatted Date"].dt.hour
    df["dayofweek"] = df["Formatted Date"].dt.dayofweek

    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe with original columns and returns
    a new dataframe with only the feature columns used by the model.
    """
    df = add_time_features(df)

    # Make sure all required columns exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = None

    return df[FEATURE_COLS]
