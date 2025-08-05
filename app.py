import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and label encoder
model = joblib.load("aqi_rf_model (1).joblib")
label_encoder = joblib.load("label_encoder (3).joblib")

# Page config
st.set_page_config(page_title="Delhi AQI Predictor 🌫️", layout="centered")
st.title("🌫️ Delhi Air Quality Index (AQI) Predictor")
st.markdown("Enter pollutant levels to predict the AQI **Category**.")

# Input layout
col1, col2 = st.columns(2)
with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=120.0)
    no2 = st.number_input("NO₂ (µg/m³)", min_value=0.0, value=40.0)
    co = st.number_input("CO (mg/m³)", min_value=0.0, value=1.2)
with col2:
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=180.0)
    so2 = st.number_input("SO₂ (µg/m³)", min_value=0.0, value=10.0)
    ozone = st.number_input("Ozone (µg/m³)", min_value=0.0, value=20.0)

# Predict button
if st.button("\ud83d\udd2e Predict AQI Category"):
    input_data = np.array([[pm25, pm10, no2, so2, co, ozone]])
    pred_encoded = model.predict(input_data)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    # Result display
    st.subheader("\ud83d\udccc Predicted AQI Category:")
    color_map = {
        "Good": "\ud83d\udfe2",
        "Satisfactory": "\ud83d\udfe1",
        "Moderate": "\ud83d\udfe0",
        "Poor": "\ud83d\udd34",
        "Very Poor": "\ud83d\udd23",
        "Severe": "\u26ab\ufe0f"
    }
    emoji = color_map.get(pred_label, "\u2753")
    st.success(f"{emoji} **{pred_label}**")

    # SHAP explanation
    st.markdown("---")
    st.markdown("\ud83d\udcca **Feature Contribution (SHAP Visualization)**")

    try:
        explainer = shap.Explainer(model, feature_names=["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"])
        shap_values = explainer(input_data)

        # Waterfall Plot
        st.markdown("\ud83d\udcc9 **Waterfall Plot for Explanation**")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig1)
        plt.clf()

        # Optional: Bar Plot
        if st.checkbox("Show SHAP Bar Plot"):
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            shap.plots.bar(shap_values, show=False)
            st.pyplot(fig2)
            plt.clf()

    except Exception as e:
        st.warning(f"\u26a0\ufe0f SHAP explanation could not be generated: {e}")

# Info section
with st.expander("\u2139\ufe0f About AQI Categories"):
    st.markdown("""
- **Good (0–50)**: Minimal impact  
- **Satisfactory (51–100)**: Minor breathing discomfort  
- **Moderate (101–200)**: Discomfort to sensitive people  
- **Poor (201–300)**: Breathing discomfort  
- **Very Poor (301–400)**: Respiratory illness  
- **Severe (401–500)**: Affects healthy people  
    """)

# Footer
st.markdown("---")
st.caption("Created by Alok Tungal | Powered by Random Forest \ud83c\udf33 + SHAP Explainability")
