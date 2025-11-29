# streamlit_app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="California Housing Predictor", layout="centered")

# --- Load model bundle ---
@st.cache_resource
def load_bundle(path="model.pkl"):
    try:
        bundle = joblib.load(path)
        model = bundle["model"]
        scaler = bundle["scaler"]
        train_columns = bundle["columns"]
        return model, scaler, train_columns
    except Exception as e:
        st.error(f"Failed to load model bundle: {e}")
        raise

model, scaler, train_columns = load_bundle("model.pkl")

st.title("🏡 California Housing Price Predictor")

st.markdown(
    "Enter property features below and click **Predict House Value**. "
    "If predictions don't change when you change `ocean_proximity`, retrain the model to include that feature."
)

# -------- Input Fields -------- #
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        longitude = st.number_input("Longitude", value=-120.0, format="%.6f")
        latitude = st.number_input("Latitude", value=35.0, format="%.6f")
        housing_median_age = st.number_input("Housing Median Age", value=20.0)
        total_rooms = st.number_input("Total Rooms", value=1000.0)
        total_bedrooms = st.number_input("Total Bedrooms", value=200.0)
    with col2:
        population = st.number_input("Population", value=800.0)
        households = st.number_input("Households", value=300.0)
        median_income = st.number_input("Median Income", value=3.0)
        ocean_proximity = st.selectbox(
            "Ocean Proximity",
            ["INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND", "<1H OCEAN"],
            index=0
        )

    debug = st.checkbox("Show debug info (input features & dummies)")
    submitted = st.form_submit_button("Predict House Value")

# -------- Prediction Logic -------- #
def preprocess_input(data_dict, train_columns):
    """
    data_dict: single-row dict of raw inputs
    train_columns: list of columns used during training (order matters)
    Returns: DataFrame with exactly train_columns (filled with zeros for missing).
    """
    df = pd.DataFrame(data_dict)

    # Fill and types
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    df['ocean_proximity'] = df['ocean_proximity'].astype('category')

    # Avoid zeros before division / log
    df['households'] = df['households'].replace(0, 1e-6)
    df['total_rooms'] = df['total_rooms'].replace(0, 1e-6)

    # Feature engineering
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_rooms'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']

    # Log transforms (same order used in training)
    cols_to_log = [
        "total_rooms", "total_bedrooms", "population", "households",
        "rooms_per_household", "bedrooms_per_rooms", "population_per_household"
    ]
    for c in cols_to_log:
        df[c] = np.log(df[c] + 1)

    # One-hot encode ocean_proximity (ensure categories become dummies)
    df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=False)

    # Reindex to training columns, fill missing with 0, and ensure order
    df = df.reindex(columns=train_columns, fill_value=0)

    return df

if submitted:
    # Prepare single-row dict
    data = {
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity],
    }

    X_df = preprocess_input(data, train_columns)

    if debug:
        st.subheader("🔎 Debug: final model input (first row)")
        st.write(X_df.head(1))
        ocean_cols = [c for c in X_df.columns if "ocean_proximity" in c]
        if ocean_cols:
            st.write("🔹 ocean_proximity dummy columns and values:")
            st.write(X_df[ocean_cols].head(1))
        else:
            st.warning("No ocean_proximity dummy columns found in training columns. The model was likely trained without that feature.")

    try:
        # Scale and predict
        X_scaled = scaler.transform(X_df)
        pred = model.predict(X_scaled)[0]

        # Show results
        # If your model's target is in dollars already, show raw. If it was scaled during training, adjust accordingly.
        st.success(f"🏠 Predicted Median House Value: ${pred:,.2f}")
        st.write("Note: Value is the same units used during model training (usually USD).")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)
