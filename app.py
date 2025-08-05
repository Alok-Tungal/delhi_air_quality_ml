import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# Load model and encoder
model = joblib.load("aqi_rf_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# Set up Streamlit page
st.set_page_config(page_title="🌫️ Delhi AQI Predictor", layout="centered")

# Custom CSS for beauty
st.markdown("""
<style>
    .main {
        background-color: #f4f9f9;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-size: 16px;
    }
    .stMarkdown h1 {
        font-size: 36px;
        color: #4A4A4A;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with option menu
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Predict AQI", "About"],
        icons=["cloud-fog2", "info-circle"],
        menu_icon="cast",
        default_index=0
    )

if selected == "Predict AQI":
    st.title("🌫️ Delhi Air Quality Index (AQI) Predictor")
    st.write("Enter pollutant levels below to predict the AQI **Category**.")

    col1, col2 = st.columns(2)
    with col1:
        pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=120.0)
        no2 = st.number_input("NO₂ (µg/m³)", min_value=0.0, value=40.0)
        co = st.number_input("CO (mg/m³)", min_value=0.0, value=1.2)
    with col2:
        pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=180.0)
        so2 = st.number_input("SO₂ (µg/m³)", min_value=0.0, value=10.0)
        ozone = st.number_input("Ozone (µg/m³)", min_value=0.0, value=20.0)

    if st.button("🔮 Predict AQI Category"):
        input_data = np.array([[pm25, pm10, no2, so2, co, ozone]])
        pred_encoded = model.predict(input_data)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        color_map = {
            "Good": "🟢",
            "Satisfactory": "🟡",
            "Moderate": "🟠",
            "Poor": "🔴",
            "Very Poor": "🟣",
            "Severe": "⚫️"
        }
        emoji = color_map.get(pred_label, "❓")
        st.markdown(f"### 📌 Predicted AQI Category: {emoji} **{pred_label}**")

        st.markdown("---")
        st.markdown("### 📊 Feature Contribution (SHAP Explanation)")
        try:
            explainer = shap.Explainer(model, feature_names=["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"])
            shap_values = explainer(input_data)

            st.markdown("📉 Waterfall Plot:")
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig1)
            plt.clf()

            if st.checkbox("Show SHAP Bar Plot"):
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                shap.plots.bar(shap_values, show=False)
                st.pyplot(fig2)
                plt.clf()

        except Exception as e:
            st.warning(f"⚠️ SHAP explanation could not be generated: {e}")

elif selected == "About":
    st.title("ℹ️ About AQI Categories")
    st.markdown("""
- **Good (0–50)**: Minimal impact  
- **Satisfactory (51–100)**: Minor breathing discomfort  
- **Moderate (101–200)**: Discomfort to sensitive people  
- **Poor (201–300)**: Breathing discomfort  
- **Very Poor (301–400)**: Respiratory illness  
- **Severe (401–500)**: Affects healthy people
    """)
    st.markdown("---")
    st.caption("Created by Alok Tungal | ✨ Designed with ❤️ + Random Forest + SHAP")
