import streamlit as st
import numpy as np
import pickle

import joblib

# Load model and label encoder with joblib
model = joblib.load('aqi_rf_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')


# 🎨 Page config
st.set_page_config(page_title="Delhi AQI Predictor 🌫️", layout="centered")

# 📌 Sidebar
with st.sidebar:
    st.title("🌫️ AQI Predictor App")
    st.markdown("Predict **Air Quality Category** for Delhi based on pollutant levels.")
    st.markdown("Created by **Alok Tungal** 💻")

# 🟢 Main Section
st.markdown("## 🔍 Enter Pollutant Data")

# 📥 Inputs
col1, col2 = st.columns(2)
with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=120.0)
    no2 = st.number_input("NO₂ (µg/m³)", min_value=0.0, value=40.0)
    co = st.number_input("CO (mg/m³)", min_value=0.0, value=1.2)
with col2:
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=180.0)
    so2 = st.number_input("SO₂ (µg/m³)", min_value=0.0, value=10.0)
    ozone = st.number_input("Ozone (µg/m³)", min_value=0.0, value=20.0)

# 📤 Prediction
if st.button("🔮 Predict AQI Category"):
    input_data = np.array([[pm25, pm10, no2, so2, co, ozone]])
    pred_encoded = model.predict(input_data)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    # 🟨 Styled Result
    st.markdown("### 📌 Predicted AQI Category:")
    color_map = {
        "Good": "🟢",
        "Satisfactory": "🟡",
        "Moderate": "🟠",
        "Poor": "🔴",
        "Very Poor": "🟣",
        "Severe": "⚫️"
    }
    emoji = color_map.get(pred_label, "❓")
    st.success(f"{emoji} **{pred_label}**")

# 📘 Info
with st.expander("ℹ️ About AQI Categories"):
    st.markdown("""
- **Good (0–50)**: Minimal impact
- **Satisfactory (51–100)**: Minor breathing discomfort
- **Moderate (101–200)**: Discomfort to sensitive people
- **Poor (201–300)**: Breathing discomfort
- **Very Poor (301–400)**: Respiratory illness
- **Severe (401–500)**: Affects healthy people
    """)

# 📜 Footer
st.markdown("---")
st.markdown("📍 Based on Delhi Air Quality Dataset | © 2025")