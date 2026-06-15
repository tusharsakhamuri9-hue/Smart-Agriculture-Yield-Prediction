import joblib
import numpy as np
import pandas as pd

# Load model and encoders
model = joblib.load("models/crop_yield_model.pkl")
encoders = joblib.load("models/crop_yield_encoders.pkl")


def predict_yield(
    area,
    crop,
    state,
    season,
    district,
    crop_year
):
    # Same transformation used during training
    area_log = np.log1p(area)

    # Mean encodings
    crop_mean = encoders["crop_mean"].get(crop, 0)
    state_mean = encoders["state_mean"].get(state, 0)
    season_mean = encoders["season_mean"].get(season, 0)
    district_mean = encoders["district_mean"].get(district, 0)

    # Create dataframe with same feature names used in training
    features = pd.DataFrame({
        "Area_log": [area_log],
        "Crop_mean": [crop_mean],
        "State_mean": [state_mean],
        "Season_mean": [season_mean],
        "District_mean": [district_mean],
        "Crop_Year": [crop_year]
    })

    # Predict log yield
    prediction_log = model.predict(features)[0]

    # Convert back from log scale
    prediction = np.expm1(prediction_log)

    return prediction