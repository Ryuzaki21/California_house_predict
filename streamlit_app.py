# streamlit_app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model bundle
bundle = joblib.load("model.pkl")
model = bundle["model"]
scaler = bundle["scaler"]
train_columns = bundle["columns"]

st.title("🏡 California Housing Price Predictor (Streamlit)")

# -------- Input Fields -------- #
longitude = st.number_input("Longitude", value=-120.0)
latitude = st.number_input("Latitude", value=35.0)
housing_median_age = st.number_input("Housing Median Age", value=20)
total_rooms = st.number_input("Total Rooms", value=1000)
total_bedrooms = st.number_input("Total Bedrooms", value=200)
population = st.number_input("Population", value=800)
households = st.number_input("Households", value=300)
median_income = st.number_input("Median Income", value=3.0)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND", "<1H OCEAN"]
)

# -------- Prediction Logic -------- #
def predict_value():
    data = {
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity]
    }

    df = pd.DataFrame(data)

    # Preprocessing (same as FastAPI)
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    df['ocean_proximity'] = df['ocean_proximity'].astype('category')

    df['households'] = df['households'].replace(0, 1e-6)
    df['total_rooms'] = df['total_rooms'].replace(0, 1e-6)

    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_rooms'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']

    cols_to_log = [
        "total_rooms","total_bedrooms","population","households",
        "rooms_per_household","bedrooms_per_rooms","population_per_household"
    ]
    for c in cols_to_log:
        df[c] = np.log(df[c] + 1)

    df = pd.get_dummies(df, drop_first=True)

    # Add missing train columns
    for col in train_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[train_columns]

    X_scaled = scaler.transform(df)
    pred = model.predict(X_scaled)[0]
    return float(pred)

# -------- Button + Result UI -------- #
if st.button("Predict House Value"):
    result = predict_value()
    st.success(f"🏠 Predicted House Value: **${result:,.2f}**")
