import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page setup
st.set_page_config(
    page_title="🌍 Smart Weather Forecast",
    layout="wide",
    page_icon="🌦️"
)

# App router
st.sidebar.title("🌐 Navigation")
selection = st.sidebar.radio("Go to", ["Dashboard", "About Project", "How it Works"])

if selection == "Dashboard":
    from dashboard import run_dashboard
    run_dashboard()

elif selection == "About Project":
    st.title("📘 About the Weather Forecast App")
    st.markdown("""
    This project uses advanced ML models (CNN, GNN, Random Forest) to forecast weather based on real-time data from Open-Meteo and OpenWeatherMap APIs.

    **Key Features:**
    - Multi-city forecasts with 3/7/10-day toggle
    - Smart alerts and travel suggestions
    - Interactive map overlays and weather visuals
    - Supabase integration for persistent storage and sharing
    - Designed with responsive, user-friendly UI for all devices

    Made with ❤️ for the Hackathon.
    """)

elif selection == "How it Works":
    st.title("🛠 How it Works")
    st.markdown("""
    1. **Data Ingestion**: Weather data is pulled using Open-Meteo & OpenWeatherMap APIs.
    2. **ML Models**: Trained using historical data (30 days) to predict next 3–10 days.
    3. **Visualization**: A custom Streamlit dashboard displays forecasts with cards, charts, and maps.
    4. **Storage**: All predictions are stored and queried via Supabase.

    Full pipeline is optimized for accuracy and real-time usability.
    """)
