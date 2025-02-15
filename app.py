import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load Models and Scaler
xgb_clf = joblib.load("crop_classifier.pkl")
xgb_reg = joblib.load("yield_predictor.pkl")
scaler = joblib.load("scaler.pkl")

# Load Crop Mapping
crop_mapping = {
    0: "Rice", 1: "Maize", 2: "Chickpea", 3: "Kidney Beans", 4: "Pigeon Peas", 5: "Moth Beans", 6: "Mung Beans",
    7: "Black Gram", 8: "Lentil", 9: "Pomegranate", 10: "Banana", 11: "Mango", 12: "Grapes", 13: "Watermelon",
    14: "Muskmelon", 15: "Apple", 16: "Orange", 17: "Papaya", 18: "Coconut", 19: "Cotton", 20: "Jute", 21: "Coffee"
}

def predict_crop_yield(features):
    scaled_features = scaler.transform([features])
    crop_label = xgb_clf.predict(scaled_features)[0]
    crop_name = crop_mapping.get(crop_label, "Unknown")
    yield_prediction = xgb_reg.predict(scaled_features)[0]
    return crop_name, round(yield_prediction, 2)

# Streamlit UI
st.set_page_config(page_title="Crop & Yield Prediction", layout="wide")
st.title("🌾 Smart Farming Assistance")

st.sidebar.header("🔢 Enter Soil & Weather Data")
n = st.sidebar.slider("Nitrogen Content (N)", 0, 150, 50)
p = st.sidebar.slider("Phosphorus Content (P)", 0, 150, 50)
k = st.sidebar.slider("Potassium Content (K)", 0, 150, 50)
temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 50.0)
ph = st.sidebar.number_input("pH Level", 0.0, 14.0, 6.5)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

if st.sidebar.button("🔍 Predict Crop & Yield"):
    # Feature Engineering (Matching Model Training)
    features = [n, p, k, temperature, humidity, ph, rainfall, n / (p + 1), p / (k + 1), k / (n + 1), humidity / (temperature + 1), rainfall * ph]
    crop, predicted_yield = predict_crop_yield(features)
    
    st.success(f"🌱 Recommended Crop: **{crop}**")
    st.info(f"📊 Estimated Yield: **{predicted_yield} tons/hectare**")
    
    # Visualization
    st.subheader("📈 Crop Yield Prediction")
    st.bar_chart(pd.DataFrame({"Yield (tons/ha)": [predicted_yield]}, index=[crop]))

st.sidebar.markdown("---")
st.sidebar.info("Developed with ❤️ using Streamlit :)")
