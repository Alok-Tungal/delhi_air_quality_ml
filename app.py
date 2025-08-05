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

# 🧠 Predict
if st.button("🔮 Predict AQI Category"):
    input_data = np.array([[pm25, pm10, no2, so2, co, ozone]])
    pred_encoded = model.predict(input_data)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    # 🟨 AQI Emoji Map
    emoji_map = {
        "Good": "🟢",
        "Satisfactory": "🟡",
        "Moderate": "🟠",
        "Poor": "🔴",
        "Very Poor": "🟣",
        "Severe": "⚫️"
    }
    emoji = emoji_map.get(pred_label, "❓")

    # ✅ Beautiful Output - Light & Dark mode compatible
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


st.markdown("### 🧪 Try a Sample AQI Scenario")
selected_category = st.selectbox(
    "Pick Target AQI Category to Auto-Fill Inputs:",
    ["-- Select --", "Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
)

preset_values = {
    "Good": [25.0, 40.0, 20.0, 5.0, 0.8, 10.0],
    "Satisfactory": [60.0, 70.0, 30.0, 8.0, 1.0, 15.0],
    "Moderate": [110.0, 150.0, 50.0, 15.0, 1.5, 25.0],
    "Poor": [180.0, 250.0, 80.0, 25.0, 2.0, 35.0],
    "Very Poor": [310.0, 400.0, 110.0, 40.0, 2.5, 60.0],
    "Severe": [420.0, 500.0, 150.0, 60.0, 3.0, 90.0]
}

# Set default values
default_values = preset_values.get(selected_category, [120.0, 180.0, 40.0, 10.0, 1.2, 20.0])


col1, col2 = st.columns(2)
with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=default_values[0], key="pm25_input")
    no2 = st.number_input("NO₂ (µg/m³)", min_value=0.0, value=default_values[2], key="no2_input")
    co = st.number_input("CO (mg/m³)", min_value=0.0, value=default_values[4], key="co_input")
with col2:
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=default_values[1], key="pm10_input")
    so2 = st.number_input("SO₂ (µg/m³)", min_value=0.0, value=default_values[3], key="so2_input")
    ozone = st.number_input("Ozone (µg/m³)", min_value=0.0, value=default_values[5], key="ozone_input")




st.markdown("#### 🔁 Choose a Preset AQI Level or Enter Custom Values")

preset_values = {
    "Good": [30, 40, 20, 5, 0.4, 10],
    "Moderate": [90, 110, 40, 10, 1.2, 30],
    "Poor": [200, 250, 90, 20, 2.0, 50],
    "Very Poor": [300, 350, 120, 30, 3.5, 70],
    "Severe": [400, 500, 150, 40, 4.5, 90],
}

selected_level = st.selectbox("Choose Preset AQI Level", list(preset_values.keys()))
default_values = preset_values[selected_level]
default_values = list(map(float, default_values))  # Fix type mismatch


col1, col2 = st.columns(2)
with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=default_values[0])
    no2 = st.number_input("NO₂ (µg/m³)", min_value=0.0, value=default_values[2])
    co = st.number_input("CO (mg/m³)", min_value=0.0, value=default_values[4])
with col2:
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=default_values[1])
    so2 = st.number_input("SO₂ (µg/m³)", min_value=0.0, value=default_values[3])
    ozone = st.number_input("Ozone (µg/m³)", min_value=0.0, value=default_values[5])



# Step 3.2 - Display entered pollution levels
st.markdown("### 📋 Your Entered Pollution Levels:")
st.info(f"""
- **PM2.5:** {pm25} µg/m³  
- **PM10:** {pm10} µg/m³  
- **NO₂:** {no2} µg/m³  
- **SO₂:** {so2} µg/m³  
- **CO:** {co} mg/m³  
- **Ozone:** {ozone} µg/m³  
""")

# Show a contextual warning or tip based on PM levels
if pm25 > 250 or pm10 > 300:
    st.warning("⚠️ High levels of PM detected. Stay indoors if possible.")
elif pm25 < 50 and pm10 < 50:
    st.success("✅ Air looks clean today! Great time for a walk.")




# Step 4 - AQI Prediction & Report
if st.button("🔮 Predict AQI Category", key="predict_button"):
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

    # Show Prediction Result
    st.markdown(f"### 📌 AQI Category: {emoji} **{pred_label}**")

    # Optional: SHAP visual can be added here (if you want)

    # 📥 Download AQI Report
    import io
    summary = f"""Delhi AQI Prediction Report
-----------------------------
📌 AQI Category: {emoji} {pred_label}
-----------------------------
Pollutant Levels:
PM2.5: {pm25} µg/m³
PM10: {pm10} µg/m³
NO₂: {no2} µg/m³
SO₂: {so2} µg/m³
CO: {co} mg/m³
Ozone: {ozone} µg/m³
"""
    buffer = io.StringIO()
    buffer.write(summary)
    buffer.seek(0)

    st.download_button(
        label="📥 Download AQI Report",
        data=buffer,
        file_name="aqi_report.txt",
        mime="text/plain",
        key="download_report"
    )
