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
