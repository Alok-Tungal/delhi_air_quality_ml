import streamlit as st
from streamlit_option_menu import option_menu

# Set page config
st.set_page_config(page_title="🌫️ Delhi AQI Dashboard", layout="wide")

# Inject custom CSS for cleaner, modern look
st.markdown("""
    <style>
        body {
            background-color: #f9f9f9;
            font-family: 'Segoe UI', sans-serif;
        }
        .main, .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            padding: 8px 20px;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #1F2937;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        menu_title="🌫️ Delhi AQI App",
        options=["Live AQI Dashboard", "Predict AQI", "AQI History", "Pollutant Info", "About"],
        icons=["cloud-fog2", "graph-up", "bar-chart-line", "info-circle", "person-circle"],
        menu_icon="cast",
        default_index=0,
    )

# Placeholder Pages (will be filled in future steps)
if selected == "Live AQI Dashboard":
    st.title("📡 Live Delhi AQI Dashboard")
    st.info("We will integrate live AQI from OpenAQ API here.")

elif selected == "Predict AQI":
    st.title("🤖 Predict AQI Category")
    st.warning("This will use your trained ML model with SHAP analysis.")

elif selected == "AQI History":
    st.title("📈 AQI History & Trends")
    st.info("Time series line chart & heatmap coming soon.")

elif selected == "Pollutant Info":
    st.title("🧪 Pollutant Information")
    st.success("Will display health impact & limits of PM2.5, NO2, etc.")

elif selected == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    **Creator**: Alok Tungal  
    **Purpose**: Predict and analyze Delhi's air quality using AI and real-time data.  
    **Tech Used**: Python, Streamlit, scikit-learn, SHAP, OpenAQ API
    """)



import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and encoder
model = joblib.load("aqi_rf_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# Page title
st.title("🔮 **Predict Delhi AQI Category**")
st.markdown("Enter the pollutant levels below to predict the **Air Quality Index (AQI)** category.")

# Input form
with st.form("aqi_form"):
    col1, col2 = st.columns(2)
    with col1:
        pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 1000.0, 120.0)
        no2 = st.number_input("NO₂ (µg/m³)", 0.0, 1000.0, 40.0)
        co = st.number_input("CO (mg/m³)", 0.0, 50.0, 1.2)
    with col2:
        pm10 = st.number_input("PM10 (µg/m³)", 0.0, 1000.0, 180.0)
        so2 = st.number_input("SO₂ (µg/m³)", 0.0, 1000.0, 10.0)
        ozone = st.number_input("Ozone (µg/m³)", 0.0, 1000.0, 20.0)

    submitted = st.form_submit_button("🔍 Predict AQI")

if submitted:
    input_data = np.array([[pm25, pm10, no2, so2, co, ozone]])
    pred_encoded = model.predict(input_data)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    emoji_map = {
        "Good": "🟢",
        "Satisfactory": "🟡",
        "Moderate": "🟠",
        "Poor": "🔴",
        "Very Poor": "🟣",
        "Severe": "⚫️"
    }
  st.success(f"📌 Predicted AQI Category: {emoji} **{pred_label}**")


    st.markdown("---")
    st.markdown("📊 **SHAP Explainability**")

    try:
        explainer = shap.Explainer(model, feature_names=["PM2.5", "PM10", "NO₂", "SO₂", "CO", "Ozone"])
        shap_values = explainer(input_data)

        if len(shap_values.values.shape) == 3:  # Multiclass
            class_index = pred_encoded
            class_shap = shap_values.values[0][class_index]

            fig1, ax1 = plt.subplots(figsize=(10, 4))
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value[class_index],
                class_shap,
                feature_names=explainer.feature_names,
                features=input_data[0]
            )
            st.pyplot(fig1)
            plt.clf()

        else:
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value,
                shap_values.values[0],
                feature_names=explainer.feature_names,
                features=input_data[0]
            )
            st.pyplot(fig1)
            plt.clf()

    except Exception as e:
        st.warning(f"⚠️ SHAP explanation failed: {e}")






