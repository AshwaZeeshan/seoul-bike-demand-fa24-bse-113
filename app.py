from pathlib import Path
from datetime import date
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from features import prepare_features

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="Seoul Bike Demand", page_icon="🚲", layout="wide")

@st.cache_resource
def load_assets():
    return joblib.load(BASE / "seoul_bike_model.pkl"), json.loads((BASE / "model_metadata.json").read_text(encoding="utf-8"))

st.title("Seoul Bike Sharing Demand Prediction")
st.write("Estimate citywide bike rentals for one hour from the calendar, weather, and service status.")
st.caption("Shifa Zeeshan | FA24-BSE-113 | Section B | Machine Learning Fundamentals, Assignment 3")
try:
    model, metadata = load_assets()
except FileNotFoundError:
    st.error("Model files are missing. Run all cells in seoul_bike_prediction.ipynb and keep the generated model and metadata beside app.py.")
    st.stop()

with st.form("bike_form"):
    left, right = st.columns(2)
    with left:
        chosen_date = st.date_input("Date", value=date(2018, 6, 15), min_value=date(2017, 12, 1), max_value=date(2030, 12, 31))
        hour = st.slider("Hour (0 = midnight)", 0, 23, 18)
        temperature = st.number_input("Temperature (°C)", min_value=-40.0, max_value=50.0, value=25.0)
        humidity = st.slider("Humidity (%)", 0, 100, 55)
        wind = st.number_input("Wind speed (m/s)", min_value=0.0, max_value=50.0, value=1.5)
        visibility = st.number_input("Visibility (10 m units)", min_value=0, max_value=5000, value=2000)
    with right:
        dew_point = st.number_input("Dew point temperature (°C)", min_value=-60.0, max_value=50.0, value=15.0)
        radiation = st.number_input("Solar radiation (MJ/m²)", min_value=0.0, max_value=10.0, value=0.5)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=0.0)
        snowfall = st.number_input("Snowfall (cm)", min_value=0.0, max_value=100.0, value=0.0)
        holiday = st.selectbox("Holiday", ["No Holiday", "Holiday"])
        functioning = st.selectbox("Functioning Day", ["Yes", "No"])
    submitted = st.form_submit_button("Predict bike demand", type="primary")

if submitted:
    if dew_point > temperature:
        st.error("Dew point cannot exceed air temperature. Please correct the weather inputs.")
    else:
        month = chosen_date.month
        season = "Winter" if month in (12, 1, 2) else "Spring" if month in (3, 4, 5) else "Summer" if month in (6, 7, 8) else "Autumn"
        raw = pd.DataFrame([{"Date": chosen_date.strftime("%d/%m/%Y"), "Hour": hour,
            "Temperature": temperature, "Humidity": humidity, "Wind speed": wind,
            "Visibility": visibility, "Dew point temperature": dew_point,
            "Solar Radiation": radiation, "Rainfall": rainfall, "Snowfall": snowfall,
            "Seasons": season, "Holiday": holiday, "Functioning Day": functioning}])
        prediction = float(np.maximum(0, model.predict(prepare_features(raw))[0]))
        st.metric("Estimated rentals during this hour", f"{prediction:,.0f} bikes")
        st.caption(f"Season derived from date: {season}. Model: {metadata['best_algorithm']}.")
        if functioning == "No":
            st.info("The service is marked non-functioning. Observed rentals were zero for such hours; the displayed statistical estimate may be above zero.")
        if chosen_date < date(2017,12,1) or chosen_date > date(2018,11,30):
            st.warning("This date is outside the 2017-2018 training period. Current demand patterns may differ.")

with st.expander("Model evaluation and scope"):
    st.write("All 8,760 rows were evaluated using the same shuffled 10-fold partitions. Predictions are clipped at zero in both evaluation and this app.")
    st.write(f"Selected model mean CV RMSE: {metadata['best_mean_rmse']:.2f} bikes/hour. This is an aggregate error metric, not an individual prediction interval.")
    st.dataframe(pd.read_csv(BASE / "results" / "model_comparison.csv"), hide_index=True)
    st.write("Random folds can be optimistic for adjacent hourly data. A separate chronological holdout is reported in the notebook. Weather inputs must be supplied for the prediction hour.")
    st.markdown("Data: [UCI Seoul Bike Sharing Demand](https://doi.org/10.24432/C5F62R), CC BY 4.0.")
