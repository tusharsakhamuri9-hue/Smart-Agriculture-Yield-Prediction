import streamlit as st
import joblib
import sys
import os

sys.path.append(os.path.abspath("src"))

from predict import predict_yield

# Load encoders
encoders = joblib.load("models/crop_yield_encoders.pkl")

# Dropdown values
crop_options = sorted(encoders["crop_mean"].keys())
state_options = sorted(encoders["state_mean"].keys())
season_options = sorted(encoders["season_mean"].keys())
district_options = sorted(encoders["district_mean"].keys())

# Page Config
st.set_page_config(
    page_title="Smart Agriculture Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

# Sidebar
st.sidebar.header("Project Information")
st.sidebar.write("Machine Learning Project")
st.sidebar.write("Model: Random Forest Regressor")
st.sidebar.write("Target: Crop Yield")

st.sidebar.header("Model Performance")
st.sidebar.success("R² Score : 0.9612")
st.sidebar.success("MAE : 0.1185")
st.sidebar.success("Algorithm : Random Forest")

st.sidebar.header("MLOps")
st.sidebar.info("MLflow Experiment Tracking")
st.sidebar.info("Model Registry Enabled")

# Title
st.title("🌾 Smart Agriculture Yield Prediction")

st.markdown(
    """
Predict crop yield using Machine Learning based on agricultural inputs.

### Features
- Crop Selection
- State Selection
- District Selection
- Season Selection
- Area Based Prediction
- MLflow Experiment Tracking
"""
)

st.subheader("Enter Crop Details")

col1, col2 = st.columns(2)

with col1:

    area = st.number_input(
        "Area",
        min_value=1.0,
        value=100.0
    )

    crop = st.selectbox(
        "Crop",
        crop_options
    )

    state = st.selectbox(
        "State",
        state_options
    )

with col2:

    season = st.selectbox(
        "Season",
        season_options
    )

    district = st.selectbox(
        "District",
        district_options
    )

    crop_year = st.number_input(
        "Crop Year",
        min_value=1997,
        max_value=2030,
        value=2015
    )

# Show Inputs
st.subheader("Selected Inputs")

st.write(f"🌱 Crop: {crop}")
st.write(f"🏛 State: {state}")
st.write(f"📍 District: {district}")
st.write(f"🌦 Season: {season}")
st.write(f"📏 Area: {area}")
st.write(f"📅 Crop Year: {crop_year}")

# Prediction
if st.button("Predict Yield"):

    prediction = predict_yield(
        area=area,
        crop=crop,
        state=state,
        season=season,
        district=district,
        crop_year=crop_year
    )

    st.success("Prediction Completed")

    col_a, col_b = st.columns(2)

    with col_a:
        st.metric(
            label="🌾 Predicted Yield (tonnes/hectare)",
            value=f"{prediction:.2f}"
        )

    with col_b:
        total_production = prediction * area
        st.metric(
            label="📦 Estimated Total Production (tonnes)",
            value=f"{total_production:.2f}"
        )

    st.info(
        f"For **{area} hectares** of **{crop}** in **{district}, {state}** "
        f"during **{season} {crop_year}**, the model predicts a yield of "
        f"**{prediction:.2f} tonnes per hectare**, giving an estimated total "
        f"production of **{total_production:.2f} tonnes**."
    )

# Footer
st.markdown("---")

st.markdown(
    """
### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Random Forest Regressor
- MLflow
- Streamlit
"""
)