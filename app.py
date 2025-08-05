import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and label encoder
model = joblib.load("aqi_rf_model (1).joblib")
label_encoder = joblib.load("label_encoder (3).joblib")

# 🎨 Page configuration
st.set_page_config(page_title="Delhi AQI Predictor 🌫️", layout="centered")
st.title("🌫️ Delhi Air Quality Index (AQI) Predictor")
st.markdown("Enter pollutant levels to predict the AQI **Category**.")

# 🧾 Input layout
col1, col2 = st.columns(2)
with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=120.0)
    no2 = st.number_input("NO₂ (µg/m³)", min_value=0.0, value=40.0)
    co = st.number_input("CO (mg/m³)", min_value=0.0, value=1.2)
with col2:
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=180.0)
    so2 = st.number_input("SO₂ (µg/m³)", min_value=0.0, value=10.0)
    ozone = st.number_input("Ozone (µg/m³)", min_value=0.0, value=20.0)

# 🟢 Prediction button
st.markdown("### 🔮 Click the button to predict AQI category:")
if st.button("Predict AQI Category"):
    input_data = np.array([[pm25, pm10, no2, so2, co, ozone]])
    pred_encoded = model.predict(input_data)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    # 🧾 Display result
    st.subheader("📌 Predicted AQI Category:")
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

    # 📊 SHAP Explanation
    st.markdown("---")
    st.markdown("📊 **Feature Contribution using SHAP**")


try:
    explainer = shap.Explainer(model, feature_names=["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"])
    shap_values = explainer(input_data)

    st.markdown("📉 Waterfall Plot:")
    
    # Convert SHAP values to expected format if needed
    if hasattr(shap_values, 'values') and shap_values.values.shape[0] == 1:
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig1)
        plt.clf()
    else:
        st.warning("⚠️ SHAP returned unexpected shape for waterfall. Try with shap_values[0] only.")
except Exception as e:
    st.warning(f"⚠️ SHAP explanation could not be generated: {e}")


# ℹ️ Information section
with st.expander("ℹ️ About AQI Categories"):
    st.markdown("""
- **Good (0–50)**: Minimal impact  
- **Satisfactory (51–100)**: Minor breathing discomfort  
- **Moderate (101–200)**: Discomfort to sensitive people  
- **Poor (201–300)**: Breathing discomfort  
- **Very Poor (301–400)**: Respiratory illness  
- **Severe (401–500)**: Affects healthy people  
    """)

# 📎 Footer
st.markdown("---")
st.caption("Created by Alok Tungal | Powered by Random Forest 🌳 + SHAP Explainability")




