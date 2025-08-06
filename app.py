import streamlit as st
from streamlit_option_menu import option_menu
import io
import seaborn as sns
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


# 🌍 Show Pollution Summary (Step 3.2)
st.markdown("### 📋 Your Entered Pollution Levels:")
st.info(f"""
- **PM2.5:** {pm25} µg/m³
- **PM10:** {pm10} µg/m³
- **NO₂:** {no2} µg/m³
- **SO₂:** {so2} µg/m³
- **CO:** {co} mg/m³
- **Ozone:** {ozone} µg/m³
""")

# 🎯 Show PM-based Air Quality Advisory
if pm25 > 250 or pm10 > 300:
    st.warning("⚠️ High levels of PM detected. Stay indoors if possible.")
elif pm25 < 50 and pm10 < 50:
    st.success("✅ Air looks clean today! Great time for a walk.")


# step 4

# Step 4: Predict AQI
if st.button("🔮 Predict AQI Category", key="predict_aqi"):
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

    # ✅ Show Prediction Result
    st.markdown(f"### 📌 AQI Category: {emoji} **{pred_label}**")

    # ✅ Step 5: Health Tips & Recommendations
    st.markdown("---")
    st.markdown("🩺 **Health Impact & Recommendations:**")

    aqi_health_tips = {
        "Good": {
            "impact": "Air quality is considered satisfactory, and air pollution poses little or no risk.",
            "tip": "Enjoy your day! It’s a great time for outdoor activities. 😊"
        },
        "Satisfactory": {
            "impact": "Air quality is acceptable. However, there may be a risk for some sensitive individuals.",
            "tip": "If you have asthma or allergies, keep medications handy. 🤧"
        },
        "Moderate": {
            "impact": "Air quality is okay for most, but may cause minor irritation to sensitive groups.",
            "tip": "Avoid intense outdoor activities. Hydrate well. 💧"
        },
        "Poor": {
            "impact": "Everyone may begin to experience health effects; sensitive individuals may experience serious effects.",
            "tip": "Limit outdoor exposure. Use a mask if necessary. 😷"
        },
        "Very Poor": {
            "impact": "Health warnings of emergency conditions. Serious effects on everyone's health.",
            "tip": "Avoid going out. Stay indoors with air filters. ❌🌫️"
        },
        "Severe": {
            "impact": "Serious health effects even for healthy people.",
            "tip": "Emergency! Remain indoors and avoid all physical exertion. 🚨"
        }
    }

    if pred_label in aqi_health_tips:
        info = aqi_health_tips[pred_label]
        st.error(f"**Impact:** {info['impact']}")
        st.info(f"**Tip:** {info['tip']}")
    else:
        st.warning("No health tips available for this AQI category.")


import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.markdown("### 📊 Compare Your Pollution Levels with Delhi Averages and WHO Safe Limits")

# Reference data
historical_avg = {
    "PM2.5": 90,
    "PM10": 160,
    "NO₂": 35,
    "SO₂": 12,
    "CO": 1.0,
    "Ozone": 25
}

who_limits = {
    "PM2.5": 25,
    "PM10": 50,
    "NO₂": 40,
    "SO₂": 20,
    "CO": 4.0,
    "Ozone": 50
}

# User inputs
pollutants = ["PM2.5", "PM10", "NO₂", "SO₂", "CO", "Ozone"]
your_values = [pm25, pm10, no2, so2, co, ozone]
delhi_avg = [historical_avg[p] for p in pollutants]
who_safe = [who_limits[p] for p in pollutants]

# Create DataFrame
df_compare = pd.DataFrame({
    "Pollutant": pollutants,
    "Your Input": your_values,
    "Delhi Avg": delhi_avg,
    "WHO Limit": who_safe
})

# Melt DataFrame for seaborn
df_melt = df_compare.melt(id_vars="Pollutant", var_name="Type", value_name="Value")

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=df_melt, x="Pollutant", y="Value", hue="Type", ax=ax)
plt.title("📉 Your Pollution Levels vs Delhi Avg vs WHO Safe Limits")
plt.ylabel("Concentration")
plt.xticks(rotation=0)
plt.grid(axis="y")

# Display in Streamlit
st.pyplot(fig)
plt.clf()

st.markdown("---")
st.markdown("### 🧠 Understand the Pollutants & Their Impact")

pollutant_info = {
    "PM2.5": {
        "emoji": "🌫️",
        "source": "Combustion engines, factories, stubble burning",
        "effect": "Can penetrate deep into lungs and enter bloodstream, causing heart and lung issues.",
    },
    "PM10": {
        "emoji": "🌪️",
        "source": "Dust, construction, roads",
        "effect": "Irritates nose, throat, and lungs. Can trigger asthma.",
    },
    "NO₂": {
        "emoji": "🛻",
        "source": "Vehicle emissions, industrial activities",
        "effect": "Aggravates respiratory diseases like asthma. Increases hospital visits.",
    },
    "SO₂": {
        "emoji": "🏭",
        "source": "Coal burning, thermal power plants",
        "effect": "Affects lungs, causes wheezing, shortness of breath.",
    },
    "CO": {
        "emoji": "🚗",
        "source": "Incomplete combustion in vehicles, stoves",
        "effect": "Reduces oxygen supply to body organs. Dangerous in enclosed areas.",
    },
    "Ozone": {
        "emoji": "☀️",
        "source": "Formed by sunlight reacting with pollutants (secondary pollutant)",
        "effect": "Causes chest pain, coughing, worsens bronchitis & asthma.",
    }
}

for pollutant, details in pollutant_info.items():
    st.markdown(f"""
**{details['emoji']} {pollutant}**
- **Source:** {details['source']}
- **Health Effect:** {details['effect']}
    """)
    
# ✅ STEP 7: AQI Knowledge Hub 🧠💨
with st.expander("📚 Learn About AQI & Health Tips"):
    st.markdown("### 💡 What Do These Pollutants Mean?")
    
    st.markdown("""
- **🟤 PM2.5 (Fine Particles):** Penetrates deep into lungs. Sources: dust, smoke.
- **🟠 PM10 (Coarse Particles):** Irritates eyes, nose, and throat.
- **🟣 NO₂ (Nitrogen Dioxide):** Increases asthma risk, especially in children.
- **🔵 SO₂ (Sulfur Dioxide):** Causes coughing, shortness of breath.
- **⚫ CO (Carbon Monoxide):** Reduces oxygen to brain; very dangerous at high levels.
- **🟢 Ozone (O₃):** Harmful at ground level — affects lung function.
""")

    st.markdown("### 📈 AQI Historical Meaning:")
    st.info("""
- AQI below **100** = Generally safe for most people.
- AQI above **200** = Can be dangerous for sensitive groups.
- AQI **above 300** = Public health emergency levels!
    """)

    st.markdown("### 🧘 Health Tips for High AQI Days:")
    st.success("""
- ✅ Stay indoors & use air purifiers
- ✅ Wear N95 masks outdoors
- ✅ Drink water to stay hydrated
- ✅ Avoid morning walks on high-pollution days
""")

    # ✅ Fixed Download Button (text string instead of StringIO)
    education_text = """
Air Quality & You 🌍

Pollutants Explained:
- PM2.5, PM10 → Lung irritants
- NO2, SO2 → Harmful to respiratory system
- CO → Oxygen blocker
- Ozone → Triggers asthma

Stay safe:
✔ Stay indoors on high AQI days
✔ Use masks, purifiers, and hydrate often

Made with ❤️ by Alok Tungal
    """
    st.download_button(
        label="📥 Download AQI Safety Guide",
        data=education_text,  # 🛠️ Send string instead of StringIO
        file_name="aqi_safety_guide.txt",
        mime="text/plain",
        key="download_guide_education"
    )

# step 6
# Step 6: Show Recent AQI Trend (Static Sample Data for Demo)

import pandas as pd
import random
st.markdown("---")
st.markdown("📈 **Recent AQI Trends (Simulated)**")

# Sample dummy data for past 7 days
trend_data = {
    "Date": pd.date_range(end=pd.Timestamp.today(), periods=7).strftime("%Y-%m-%d"),
    "AQI": [random.randint(80, 450) for _ in range(7)]
}
df_trend = pd.DataFrame(trend_data)

# Plot the AQI line chart
st.line_chart(df_trend.set_index("Date"), use_container_width=True)

# Add a mini table below
st.dataframe(df_trend.rename(columns={"Date": "📅 Date", "AQI": "🌫️ AQI Value"}), use_container_width=True)


import qrcode
from PIL import Image
import streamlit as st
from io import BytesIO
import urllib.parse
import os

input_data = np.array([[pm25, pm10, no2, so2, co, ozone]])
pred_encoded = model.predict(input_data)[0]
pred_label = label_encoder.inverse_transform([pred_encoded])[0]
paste_url = "https://alokdelhiairqualityml.streamlit.app/"

color_map = {
    "Good": "🟢",
    "Satisfactory": "🟡",
    "Moderate": "🟠",
    "Poor": "🔴",
    "Very Poor": "🟣",
    "Severe": "⚫️"
}
emoji = color_map.get(pred_label, "❓")

# ✅ Optional social media share
tweet_text = f"Delhi AQI today is {pred_label} {emoji}. Check pollution levels here: {paste_url} #AQI #AirQuality"
tweet_url = f"https://twitter.com/intent/tweet?text={urllib.parse.quote(tweet_text)}"

st.markdown("### 📤 Share on Social Media")
st.markdown(f"[🐦 Tweet This Report]({tweet_url})", unsafe_allow_html=True)

# Generate QR Code with high box_size for clarity
qr = qrcode.QRCode(
    version=1,
    box_size=10,
    border=4
)
qr.add_data(paste_url)
qr.make(fit=True)

# Create and resize image for laptop viewing
img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
img = img.resize((300, 300), Image.LANCZOS)  # Clear and sharp

# Display QR code with updated Streamlit parameter
st.image(img, caption="📱 Scan to open the report", use_container_width=False)

# ✅ Show in Streamlit
st.markdown("### 📲 Share This AQI Summary via QR Code")
# st.image(qr_path, caption="🔗 Scan to open AQI Report", use_container_width=True)

# Optional: Download QR Code
buf = BytesIO()
img.save(buf, format="PNG")
byte_im = buf.getvalue()

st.download_button(
    label="📥 Download QR Code",
    data=byte_im,
    file_name="Delhi_AQI_QR_Code.png",
    mime="image/png"
)


inputs = {
    "PM2.5": pm25,
    "PM10": pm10,
    "NO2": no2,
    "SO2": so2,
    "CO": co,
    "Ozone": ozone
}

# 4. 🧠 Use preset label as AQI category (simulate ML prediction here)
aqi_category = selected_level  # You can replace this with your ML model's output if needed

# 5. 🚦 Risk Badge Generator
def get_risk_badge(aqi_category, inputs):
    main_pollutant = max(inputs, key=inputs.get)
    risk = {
        "Good": "LOW",
        "Satisfactory": "LOW",
        "Moderate": "MEDIUM",
        "Poor": "HIGH",
        "Very Poor": "HIGH",
        "Severe": "CRITICAL"
    }.get(aqi_category, "UNKNOWN")

    emoji = {
        "LOW": "🟢",
        "MEDIUM": "🟠",
        "HIGH": "🔴",
        "CRITICAL": "🚨"
    }.get(risk, "❓")

    return main_pollutant, risk, emoji

# 6. 🎯 Display Summary
main_pollutant, risk, emoji = get_risk_badge(aqi_category, inputs)

st.markdown("---")
st.markdown(f"""
### {emoji} Pollution Risk Summary
- **Risk Level:** `{risk}`
- **Main Pollutant:** `{main_pollutant}`
- **AQI Category:** `{aqi_category}`
""")
st.markdown("---")


import streamlit as st
import qrcode
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# ---------------- Page Setup ----------------
st.set_page_config(
    page_title="Delhi AQI Predictor",
    layout="centered",  # 'wide' is optional, centered is better for mobile
    initial_sidebar_state="expanded"
)
st.markdown("<div id='top'></div>", unsafe_allow_html=True)
st.title("🏙️ Delhi AQI Predictor")
st.markdown("Mobile-friendly | Streamlined | Responsive")

# ---------------- Preset AQI Values ----------------
st.markdown("### 🔁 Choose Preset or Enter Pollution Data")
preset_values = {
    "Good": [30, 40, 20, 5, 0.4, 10],
    "Moderate": [90, 110, 40, 10, 1.2, 30],
    "Poor": [200, 250, 90, 20, 2.0, 50],
    "Very Poor": [300, 350, 120, 30, 3.5, 70],
    "Severe": [400, 500, 150, 40, 4.5, 90],
}
selected_level = st.selectbox("📦 Choose Preset AQI Level", list(preset_values.keys()))
default_values = list(map(float, preset_values[selected_level]))

# ---------------- Inputs (Mobile Optimized) ----------------
st.markdown("#### 🧪 Enter Pollutant Levels")

st.markdown("**PM Pollutants**")
pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=default_values[0])
pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=default_values[1])

st.markdown("**Gaseous Pollutants**")
no2 = st.number_input("NO₂ (µg/m³)", min_value=0.0, value=default_values[2])
so2 = st.number_input("SO₂ (µg/m³)", min_value=0.0, value=default_values[3])
co = st.number_input("CO (mg/m³)", min_value=0.0, value=default_values[4])
ozone = st.number_input("Ozone (µg/m³)", min_value=0.0, value=default_values[5])

inputs = {
    "PM2.5": pm25,
    "PM10": pm10,
    "NO2": no2,
    "SO2": so2,
    "CO": co,
    "Ozone": ozone
}

# ---------------- AQI Category Simulated ----------------
aqi_category = selected_level  # Replace with model.predict() if needed

# ---------------- Risk Badge ----------------
def get_risk_badge(aqi_category, inputs):
    main_pollutant = max(inputs, key=inputs.get)
    risk = {
        "Good": "LOW",
        "Satisfactory": "LOW",
        "Moderate": "MEDIUM",
        "Poor": "HIGH",
        "Very Poor": "HIGH",
        "Severe": "CRITICAL"
    }.get(aqi_category, "UNKNOWN")
    emoji = {
        "LOW": "🟢",
        "MEDIUM": "🟠",
        "HIGH": "🔴",
        "CRITICAL": "🚨"
    }.get(risk, "❓")
    return main_pollutant, risk, emoji

main_pollutant, risk, emoji = get_risk_badge(aqi_category, inputs)

st.markdown("---")
st.markdown(f"""
### {emoji} Pollution Risk Summary
- **Risk Level:** `{risk}`
- **Main Pollutant:** `{main_pollutant}`
- **AQI Category:** `{aqi_category}`
""")
st.markdown("---")

# ---------------- SHAP Plot Placeholder ----------------
with st.expander("🔍 SHAP Model Explanation (Optional Placeholder)"):
    st.info("Insert your SHAP plots here...")

# ---------------- QR Code Section ----------------
st.markdown("### 📲 Share Report via QR Code")
paste_url = "https://your_link_here.com"
qr = qrcode.QRCode(version=1, box_size=10, border=4)
qr.add_data(paste_url)
qr.make(fit=True)
img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
img = img.resize((250, 250), Image.LANCZOS)

with st.container():
    st.image(img, caption="📷 Scan to View", use_container_width=False)

buf = BytesIO()
img.save(buf, format="PNG")
st.download_button("📥 Download QR Code", data=buf.getvalue(), file_name="Delhi_AQI_QR.png", mime="image/png")

# ---------------- Footer Scroll to Top ----------------
st.markdown("---")
st.markdown("""
<a href="#top" style="text-decoration: none;">⬆️ **Back to Top**</a>
""", unsafe_allow_html=True)


