# src/train_model.py

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# If you run as: python -m src.train_model
from preprocess import build_feature_frame, NUM_COLS, CAT_COLS
# If you run as: python src/train_model.py directly, and this import fails,
# change the above line to: from preprocess import build_feature_frame, NUM_COLS, CAT_COLS

DATA_PATH = os.path.join("data", "weatherHistory.csv")
MODEL_PATH = os.path.join("models", "weather_model.joblib")


def main():
    os.makedirs("models", exist_ok=True)

    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    print("Loaded data:", df.shape)

    # 2. Features and target
    X = build_feature_frame(df)
    y = df["Temperature (C)"]

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Preprocessing for numeric and categorical columns
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUM_COLS),
            ("cat", categorical_transformer, CAT_COLS),
        ]
    )

    # 5. Linear Regression model
    model = LinearRegression()

    # 6. Full pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    # 7. Train
    print("Training Linear Regression model...")
    pipeline.fit(X_train, y_train)

    # 8. Evaluate
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")

    # 9. Save the trained pipeline
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
