import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("data/crop_production.csv")

# =====================================================
# Feature Engineering
# =====================================================

df["Yield"] = df["Production"] / df["Area"]

# Remove invalid values
df["Yield"] = df["Yield"].replace([np.inf, -np.inf], np.nan)

# Drop invalid rows
df = df.dropna(subset=["Yield"])

# Log Transformations
df["Yield_log"] = np.log1p(df["Yield"])
df["Area_log"] = np.log1p(df["Area"])

# Remove remaining NaN rows
df = df.dropna(subset=["Yield_log", "Area_log"])

print("Dataset Shape:", df.shape)
print("NaN in Yield_log:", df["Yield_log"].isna().sum())

# =====================================================
# Target & Features
# =====================================================

target = "Yield_log"

X_raw = df[
    [
        "Area_log",
        "Crop",
        "State_Name",
        "Season",
        "District_Name",
        "Crop_Year"
    ]
]

y = df[target]

# =====================================================
# Train Test Split
# =====================================================

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# Mean Encoding
# =====================================================

train_with_target = X_train_raw.copy()
train_with_target[target] = y_train

crop_mean = train_with_target.groupby("Crop")[target].mean()
state_mean = train_with_target.groupby("State_Name")[target].mean()
season_mean = train_with_target.groupby("Season")[target].mean()
district_mean = train_with_target.groupby("District_Name")[target].mean()

encoders = {
    "crop_mean": crop_mean.to_dict(),
    "state_mean": state_mean.to_dict(),
    "season_mean": season_mean.to_dict(),
    "district_mean": district_mean.to_dict()
}

# =====================================================
# Encode Train Data
# =====================================================

X_train = pd.DataFrame({
    "Area_log": X_train_raw["Area_log"],
    "Crop_mean": X_train_raw["Crop"].map(crop_mean),
    "State_mean": X_train_raw["State_Name"].map(state_mean),
    "Season_mean": X_train_raw["Season"].map(season_mean),
    "District_mean": X_train_raw["District_Name"].map(district_mean),
    "Crop_Year": X_train_raw["Crop_Year"]
})

# =====================================================
# Encode Test Data
# =====================================================

X_test = pd.DataFrame({
    "Area_log": X_test_raw["Area_log"],
    "Crop_mean": X_test_raw["Crop"].map(crop_mean),
    "State_mean": X_test_raw["State_Name"].map(state_mean),
    "Season_mean": X_test_raw["Season"].map(season_mean),
    "District_mean": X_test_raw["District_Name"].map(district_mean),
    "Crop_Year": X_test_raw["Crop_Year"]
})

# Fill unseen categories
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# =====================================================
# Model
# =====================================================

final_model = RandomForestRegressor(
    n_estimators=246,
    max_depth=17,
    min_samples_split=20,
    min_samples_leaf=4,
    random_state=42
)

# =====================================================
# Training
# =====================================================

final_model.fit(X_train, y_train)

# =====================================================
# Evaluation
# =====================================================

y_pred = final_model.predict(X_test)

train_score = final_model.score(X_train, y_train)
test_score = final_model.score(X_test, y_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# =====================================================
# MLflow Tracking
# =====================================================

mlflow.set_experiment("Crop Yield Prediction")

with mlflow.start_run():

    # Parameters
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 246)
    mlflow.log_param("max_depth", 17)
    mlflow.log_param("min_samples_split", 20)
    mlflow.log_param("min_samples_leaf", 4)
    mlflow.log_param("random_state", 42)

    # Dataset Info
    mlflow.log_param("dataset_rows", len(df))
    mlflow.log_param("dataset_columns", len(df.columns))

    # Metrics
    mlflow.log_metric("train_score", train_score)
    mlflow.log_metric("test_score", test_score)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)

    # Model
    mlflow.sklearn.log_model(
        final_model,
        "crop_yield_model"
    )

    # Artifacts
    mlflow.log_artifact("models/crop_yield_model.pkl")
    mlflow.log_artifact("models/crop_yield_encoders.pkl")

# =====================================================
# Save Files
# =====================================================

joblib.dump(final_model, "models/crop_yield_model.pkl")
joblib.dump(encoders, "models/crop_yield_encoders.pkl")

# =====================================================
# Results
# =====================================================

print("\nTraining Complete")
print("Train Score :", train_score)
print("Test Score  :", test_score)
print("R2 Score    :", r2)
print("MAE         :", mae)
print("RMSE        :", rmse)
print("MLflow Tracking Complete")