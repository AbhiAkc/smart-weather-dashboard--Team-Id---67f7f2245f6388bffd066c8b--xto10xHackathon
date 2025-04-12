import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import plotly.express as px
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from functools import lru_cache
import requests_cache
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import base64
from gtts import gTTS
import time

# Setup cache for API calls
requests_cache.install_cache("weather_cache", expire_after=3600)

# Load environment variables and initialize Supabase
load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Geolocation setup
geolocator = Nominatim(user_agent="weather_dashboard")

@lru_cache(maxsize=1000)
def geocode_city(city_name):
    try:
        location = geolocator.geocode(city_name)
        return location.latitude, location.longitude if location else (None, None)
    except:
        return None, None

# Define cities with coordinates
cities = {
    "Delhi": {"lat": 28.7041, "lon": 77.1025},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639},
    "London": {"lat": 51.5074, "lon": -0.1278},
    "New York": {"lat": 40.7128, "lon": -74.0060},
    "Tokyo": {"lat": 35.6762, "lon": 139.6503},
    "Los Angeles": {"lat": 34.0522, "lon": -118.2437},
    "Seattle": {"lat": 47.6062, "lon": -122.3321},
    "Rabat": {"lat": 34.0202, "lon": -6.8299},
    "Manila": {"lat": 14.5995, "lon": 120.9842},
    "Istanbul": {"lat": 41.0082, "lon": 28.9784},
    "Paris": {"lat": 48.8566, "lon": 2.3522},
    "Sydney": {"lat": -33.8688, "lon": 151.2093},
}

# Weather codes mapping with icons
weather_codes = {
    0: {"description": "Clear sky", "icon_url": "https://openweathermap.org/img/wn/01d.png"},
    1: {"description": "Mainly clear", "icon_url": "https://openweathermap.org/img/wn/02d.png"},
    2: {"description": "Partly cloudy", "icon_url": "https://openweathermap.org/img/wn/03d.png"},
    3: {"description": "Overcast", "icon_url": "https://openweathermap.org/img/wn/04d.png"},
    45: {"description": "Fog", "icon_url": "https://openweathermap.org/img/wn/50d.png"},
    48: {"description": "Depositing rime fog", "icon_url": "https://openweathermap.org/img/wn/50d.png"},
    51: {"description": "Light drizzle", "icon_url": "https://openweathermap.org/img/wn/09d.png"},
    53: {"description": "Moderate drizzle", "icon_url": "https://openweathermap.org/img/wn/09d.png"},
    55: {"description": "Dense drizzle", "icon_url": "https://openweathermap.org/img/wn/09d.png"},
    61: {"description": "Slight rain", "icon_url": "https://openweathermap.org/img/wn/10d.png"},
    63: {"description": "Moderate rain", "icon_url": "https://openweathermap.org/img/wn/10d.png"},
    65: {"description": "Heavy rain", "icon_url": "https://openweathermap.org/img/wn/10d.png"},
    71: {"description": "Slight snow fall", "icon_url": "https://openweathermap.org/img/wn/13d.png"},
    73: {"description": "Moderate snow fall", "icon_url": "https://openweathermap.org/img/wn/13d.png"},
    75: {"description": "Heavy snow fall", "icon_url": "https://openweathermap.org/img/wn/13d.png"},
    80: {"description": "Slight rain showers", "icon_url": "https://openweathermap.org/img/wn/09d.png"},
    81: {"description": "Moderate rain showers", "icon_url": "https://openweathermap.org/img/wn/09d.png"},
    82: {"description": "Violent rain showers", "icon_url": "https://openweathermap.org/img/wn/09d.png"},
    85: {"description": "Slight snow showers", "icon_url": "https://openweathermap.org/img/wn/13d.png"},
    86: {"description": "Heavy snow showers", "icon_url": "https://openweathermap.org/img/wn/13d.png"},
    95: {"description": "Thunderstorm", "icon_url": "https://openweathermap.org/img/wn/11d.png"},
    96: {"description": "Thunderstorm with slight hail", "icon_url": "https://openweathermap.org/img/wn/11d.png"},
    99: {"description": "Thunderstorm with heavy hail", "icon_url": "https://openweathermap.org/img/wn/11d.png"},
}

# Page configuration
st.set_page_config(page_title="Interactive Weather Dashboard", page_icon="🌦️", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .animate-card {
        background: linear-gradient(135deg, #e0e7ff, #f0f2f6);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        animation: fadeIn 0.5s ease-in;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .animate-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .alert-card {
        background: #ffebee;
        border-left: 5px solid #d32f2f;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        animation: fadeIn 0.5s ease-in;
    }
    .forecast-card {
        background: linear-gradient(135deg, #e0e7ff, #f0f2f6);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: left;
        width: 100%;
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    @media (max-width: 600px) {
        .animate-card {
            padding: 15px;
            font-size: 0.9em;
        }
        .stColumn {
            flex: 100% !important;
            max-width: 100% !important;
        }
        h3, h4 {
            font-size: 1.2em;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Theme toggle
theme = st.sidebar.selectbox("Theme", ["Dark", "Light"], index=0)
if theme == "Dark":
    st.markdown(
        """
        <style>
        body, .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .stMarkdown, .stText {
            color: #FFFFFF;
        }
        .animate-card {
            background: linear-gradient(135deg, #2E2E2E, #3E3E2E) !important;
        }
        .alert-card {
            background-color: #4B2E2E !important;
        }
        .forecast-card {
            background: linear-gradient(135deg, #2E2E2E, #3E3E2E) !important;
            color: #FFFFFF;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Sidebar: City selection, multi-city, model selection, feedback
st.sidebar.subheader("🌍 City Selection")
city_input = st.sidebar.text_input("Search for a city")
if city_input:
    lat, lon = geocode_city(city_input)
    if lat and lon:
        cities[city_input] = {"lat": lat, "lon": lon}
        selected_city = city_input
    else:
        st.sidebar.error("City not found")
        selected_city = st.sidebar.selectbox("Or select a city", sorted(cities.keys()))
else:
    selected_city = st.sidebar.selectbox("Select a city", sorted(cities.keys()))

st.sidebar.subheader("Compare Cities")
multi_cities = st.sidebar.multiselect(
    "Select cities for comparison",
    sorted(cities.keys()),
    default=["Delhi", "Mumbai", "Bangalore", "London"],
    max_selections=4
)

st.sidebar.subheader("Model Selection")
model_type = st.sidebar.radio("Prediction Model", ["Linear Regression", "LSTM"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Feedback")
if st.sidebar.button("👍 Like this dashboard?"):
    st.sidebar.success("Thanks for your feedback! We're glad you like it!")

# Historical data fetching
def fetch_historical_data(lat, lon):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "temperature_unit": "fahrenheit",
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if not data.get("daily") or not data["daily"].get("time") or not data["daily"].get("temperature_2m_max"):
            st.warning(f"No valid historical data for coordinates ({lat}, {lon})")
            return None
        return data
    except Exception as e:
        st.warning(f"Error fetching historical data: {e}")
        return None

# Linear regression model
@st.cache_data(ttl=86400)
def train_linear_model(city, lat, lon):
    data = fetch_historical_data(lat, lon)
    if not data:
        return None

    # Create DataFrame
    df = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "temp_max": data["daily"]["temperature_2m_max"],
        "temp_min": data["daily"]["temperature_2m_min"],
        "precipitation": data["daily"]["precipitation_sum"],
    })

    # Add day of year
    df["day_of_year"] = df["date"].dt.dayofyear

    # Clean data: Handle NaN, None, or non-numeric values
    df["temp_max"] = pd.to_numeric(df["temp_max"], errors="coerce")
    df["temp_max"] = df["temp_max"].interpolate(method="linear", limit_direction="both").fillna(df["temp_max"].mean())
    df = df.dropna(subset=["temp_max", "day_of_year"])
    if df.empty:
        st.warning(f"No valid temperature data for {city} after cleaning")
        return None

    # Ensure finite values
    if not np.all(np.isfinite(df["temp_max"])):
        st.warning(f"Invalid temperature values for {city}")
        return None

    # Prepare data
    X = df[["day_of_year"]].values
    y_max = df["temp_max"].values

    # Train model
    model = LinearRegression()
    try:
        model.fit(X, y_max)
    except Exception as e:
        st.warning(f"Error training linear model for {city}: {e}")
        return None

    # Generate predictions
    last_day = df["day_of_year"].iloc[-1]
    future_days = np.array([[last_day + i] for i in range(1, 8)])
    predictions = model.predict(future_days)

    # Store predictions with retry logic
    for i, pred in enumerate(predictions):
        for attempt in range(3):
            try:
                supabase.table("weather_predictions").insert({
                    "city": city,
                    "date_time": (datetime.now() + timedelta(days=i+1)).isoformat(),
                    "temperature": float(pred),
                    "humidity": 50.0,
                    "wind_speed": 10.0,
                    "wind_direction": 0.0,
                    "precipitation": 0.0,
                    "cloud_coverage": 50.0,
                    "pressure": 1013.0,
                    "weather_condition": "Predicted"
                }).execute()
                break
            except Exception as e:
                if attempt == 2:
                    st.warning(f"Failed to store prediction for {city}, day {i+1}: {e}")
                time.sleep(1)

    return predictions

# LSTM model
@st.cache_data(ttl=86400)
def train_lstm_model(city, lat, lon):
    data = fetch_historical_data(lat, lon)
    if not data:
        return None

    # Create DataFrame
    df = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "temp_max": data["daily"]["temperature_2m_max"],
    })

    # Clean data
    df["temp_max"] = pd.to_numeric(df["temp_max"], errors="coerce")
    df["temp_max"] = df["temp_max"].interpolate(method="linear", limit_direction="both").fillna(df["temp_max"].mean())
    df = df.dropna(subset=["temp_max"])
    if df.empty:
        st.warning(f"No valid temperature data for {city}")
        return None

    # Prepare data for LSTM
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[["temp_max"]])
    X, y = [], []
    look_back = 7
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)
    if len(X) < 1:
        st.warning(f"Insufficient data for LSTM in {city}")
        return None

    # Build and train model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    try:
        model.fit(X, y, epochs=5, batch_size=1, verbose=0)  # Reduced epochs for speed
    except Exception as e:
        st.warning(f"Error training LSTM model for {city}: {e}")
        return None

    # Generate predictions
    last_sequence = scaled_data[-look_back:].reshape(1, look_back, 1)
    predictions = []
    for _ in range(7):
        pred = model.predict(last_sequence, verbose=0)
        predictions.append(pred[0, 0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[0, -1, 0] = pred
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    # Store predictions with retry logic
    for i, pred in enumerate(predictions):
        for attempt in range(3):
            try:
                supabase.table("weather_predictions").insert({
                    "city": city,
                    "date_time": (datetime.now() + timedelta(days=i+1)).isoformat(),
                    "temperature": float(pred),
                    "humidity": 50.0,
                    "wind_speed": 10.0,
                    "wind_direction": 0.0,
                    "precipitation": 0.0,
                    "cloud_coverage": 50.0,
                    "pressure": 1013.0,
                    "weather_condition": "Predicted (LSTM)"
                }).execute()
                break
            except Exception as e:
                if attempt == 2:
                    st.warning(f"Failed to store LSTM prediction for {city}, day {i+1}: {e}")
                time.sleep(1)

    return predictions

# Fetch current weather
@st.cache_data(ttl=7200)
def get_current_weather(lat, lon):
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,pressure_msl,cloud_cover,wind_speed_10m,wind_direction_10m,visibility,weather_code",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,sunrise,sunset",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "auto",
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.warning(f"Error fetching current weather: {e}")
        return None

# Fetch multi-city data
def fetch_multi_city(cities_subset):
    results = {}
    with ThreadPoolExecutor() as executor:
        futures = {city: executor.submit(get_current_weather, cities[city]["lat"], cities[city]["lon"]) for city in cities_subset}
        for city in cities_subset:
            try:
                results[city] = futures[city].result()
            except Exception as e:
                st.warning(f"Failed to fetch data for {city}: {e}")
                results[city] = None
    return results

# Weather alerts
def get_alerts(current):
    alerts = []
    if not current:
        return alerts
    temp = current.get("temperature_2m", 0)
    wind_speed = current.get("wind_speed_10m", 0)
    weather_code = current.get("weather_code", 0)
    if temp > 95:
        alerts.append({"message": "Extreme Heat Warning: Avoid outdoor activities.", "severity": "high", "color": "#ff4d4d"})
    elif temp > 85:
        alerts.append({"message": "High Temperature: Stay hydrated.", "severity": "medium", "color": "#ffcc00"})
    if wind_speed > 30:
        alerts.append({"message": "Strong Winds: Secure loose objects.", "severity": "medium", "color": "#ffcc00"})
    if weather_code in [95, 96, 99]:
        alerts.append({"message": "Thunderstorm Alert: Seek shelter.", "severity": "high", "color": "#ff4d4d"})
    if weather_code in [71, 73, 75, 85, 86]:
        alerts.append({"message": "Snow Alert: Watch for icy surfaces.", "severity": "medium", "color": "#00b7eb"})
    return alerts

# Interactive map
def display_map(lat, lon, city, temp, desc):
    try:
        m = folium.Map(location=[lat, lon], zoom_start=10)
        folium.Marker(
            [lat, lon],
            popup=f"{city}: {temp}°F, {desc}",
            tooltip=city
        ).add_to(m)
        st_folium(m, width="100%", height=300)
    except Exception as e:
        st.warning(f"Error rendering map for {city}: {e}")

# Fetch predictions from Supabase
@st.cache_data
def get_city_predictions(city):
    try:
        response = supabase.table("weather_predictions").select("*").eq("city", city).order("date_time", asc=False).execute()
        df = pd.DataFrame(response.data)
        if not df.empty:
            df['date_time'] = pd.to_datetime(df['date_time'])
            df['date_time_str'] = df['date_time'].dt.strftime('%I:%M %p, %b %d')
        return df
    except Exception as e:
        st.warning(f"Error fetching predictions for {city}: {e}")
        return pd.DataFrame()

# Text-to-speech
def text_to_speech(text):
    try:
        tts = gTTS(text)
        audio_file = "temp_audio.mp3"
        tts.save(audio_file)
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        st.audio(f"data:audio/mp3;base64,{audio_b64}")
        os.remove(audio_file)
    except Exception as e:
        st.warning(f"Error generating audio: {e}")

# 7-Day Forecast Data (Hardcoded from image)
forecast_data = [
    {"date": "Apr 12, Sat", "condition": "Slight rain showers", "high": 31.0, "low": 20.4, "sunrise": "06:08 AM", "sunset": "06:31 PM"},
    {"date": "Apr 13, Sun", "condition": "Slight rain showers", "high": 32.0, "low": 21.1, "sunrise": "06:07 AM", "sunset": "06:31 PM"},
    {"date": "Apr 14, Mon", "condition": "Overcast", "high": 32.0, "low": 22.0, "sunrise": "06:07 AM", "sunset": "06:32 PM"},
    {"date": "Apr 15, Tue", "condition": "Slight rain showers", "high": 32.7, "low": 21.9, "sunrise": "06:06 AM", "sunset": "06:32 PM"},
    {"date": "Apr 16, Wed", "condition": "Slight rain showers", "high": 31.8, "low": 21.6, "sunrise": "06:06 AM", "sunset": "06:32 PM"},
    {"date": "Apr 17, Thu", "condition": "Slight rain showers", "high": 32.1, "low": 21.7, "sunrise": "06:05 AM", "sunset": "06:32 PM"},
    {"date": "Apr 18, Fri", "condition": "Thunderstorm", "high": 32.2, "low": 22.3, "sunrise": "06:05 AM", "sunset": "06:32 PM"},
]

# Main dashboard
lat, lon = cities[selected_city]["lat"], cities[selected_city]["lon"]

# Train model
if model_type == "LSTM":
    predictions = train_lstm_model(selected_city, lat, lon)
else:
    predictions = train_linear_model(selected_city, lat, lon)

with st.spinner("Fetching weather data..."):
    weather_data = get_current_weather(lat, lon)

st.title(f"🌦️ Weather Dashboard - {selected_city}")

if weather_data:
    current = weather_data.get("current", {})
    daily = weather_data.get("daily", {})

    # Define temp and desc early
    weather_code = current.get("weather_code", 0)
    desc = weather_codes.get(weather_code, {"description": "Unknown"})["description"]
    temp = current.get("temperature_2m", "N/A")

    # Alerts
    alerts = get_alerts(current)
    for alert in alerts:
        st.markdown(
            f"""
            <div class='alert-card' style='background-color:{alert["color"]};'>
                ⚠️ <b>{alert["severity"].capitalize()} Severity</b>: {alert["message"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Main layout
    col1, col2 = st.columns([3, 2])

    with col1:
        # Current weather
        st.subheader("Current Weather")
        apparent_temp = current.get("apparent_temperature", "N/A")
        max_temp = daily.get("temperature_2m_max", [None])[0] or "N/A"
        min_temp = daily.get("temperature_2m_min", [None])[0] or "N/A"
        icon_url = weather_codes.get(weather_code, {"icon_url": ""})["icon_url"]
        st.markdown(
            f"""
            <div class='animate-card' title='Temperature: {temp}°F\nCondition: {desc}\nFeels like: {apparent_temp}°F'>
                <img src='{icon_url}' width='50'>
                <h3>{temp}°F</h3>
                <p><b>{desc}</b></p>
                <p>Feels like: {apparent_temp}°F</p>
                <p>High: {max_temp}°F | Low: {min_temp}°F</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Weather details
        st.subheader("Weather Details")
        details = [
            f"**Visibility:** {current.get('visibility', 0)/1000:.1f} km",
            f"**Dew Point:** {current.get('dew_point_2m', 'N/A')}°F",
            f"**Wind:** {current.get('wind_speed_10m', 'N/A')} mph from {current.get('wind_direction_10m', 'N/A')}°",
            f"**Humidity:** {current.get('relative_humidity_2m', 'N/A')}%",
            f"**Cloudiness:** {current.get('cloud_cover', 'N/A')}%",
            f"**Pressure:** {current.get('pressure_msl', 'N/A')} hPa",
        ]
        for detail in details:
            detail_name = detail.split(":")[0].strip("**")
            st.markdown(
                f"<div class='animate-card' title='{detail_name} for {selected_city}'>{detail}</div>",
                unsafe_allow_html=True,
            )

        # Activity suggestions
        st.subheader("Activity Suggestions")
        suggestions = []
        temp_num = float(temp) if temp != "N/A" else 0
        if temp_num > 85:
            suggestions.append("🥤 Stay hydrated and avoid strenuous outdoor activities.")
        if weather_code in [61, 63, 65, 80, 81, 82]:
            suggestions.append("☔ Carry an umbrella or raincoat.")
        if weather_code in [71, 73, 75, 85, 86]:
            suggestions.append("🧥 Dress warmly and watch for icy surfaces.")
        if not suggestions:
            suggestions.append("🏞️ Great day for outdoor activities like walking or cycling!")
        for suggestion in suggestions:
            st.markdown(
                f"<div class='animate-card' title='Activity suggestion'>{suggestion}</div>",
                unsafe_allow_html=True,
            )

    with col2:
        # Sunrise & Sunset
        st.subheader("Sunrise & Sunset")
        sunrise = datetime.fromisoformat(daily.get('sunrise', [''])[0]).strftime('%I:%M %p') if daily.get('sunrise') else "N/A"
        sunset = datetime.fromisoformat(daily.get('sunset', [''])[0]).strftime('%I:%M %p') if daily.get('sunset') else "N/A"
        st.markdown(
            f"""
            <div class='animate-card' title='Sunrise time for {selected_city}'>
                <h4><span style='font-size:24px;'>🌅</span> Sunrise</h4>
                <p>{sunrise}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class='animate-card' title='Sunset time for {selected_city}'>
                <h4><span style='font-size:24px;'>🌇</span> Sunset</h4>
                <p>{sunset}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Weather map
        st.subheader("Weather Map")
        if temp != "N/A" and desc:
            display_map(lat, lon, selected_city, temp, desc)
        else:
            st.warning("Cannot display map: Weather data unavailable.")

    # Multi-city overview
    st.subheader("Multi-City Overview")
    if multi_cities:
        cols = st.columns(len(multi_cities))
        multi_city_data = fetch_multi_city(multi_cities)
        for i, city in enumerate(multi_cities):
            with cols[i]:
                weather_data_other = multi_city_data.get(city)
                if weather_data_other and weather_data_other.get("current"):
                    current_other = weather_data_other["current"]
                    daily_other = weather_data_other.get("daily", {})
                    temp = current_other.get("temperature_2m", "N/A")
                    code = current_other.get("weather_code", 0)
                    desc = weather_codes.get(code, {"description": "Unknown"})["description"]
                    icon_url = weather_codes.get(code, {"icon_url": ""})["icon_url"]
                    max_temp = daily_other.get("temperature_2m_max", [None])[0] or "N/A"
                    min_temp = daily_other.get("temperature_2m_min", [None])[0] or "N/A"
                    st.markdown(
                        f"""
                        <div class='animate-card' title='{city}: {temp}°F, {desc}'>
                            <h4>{city}</h4>
                            <img src='{icon_url}' width='40'>
                            <p>{temp}°F</p>
                            <p>{desc}</p>
                            <p>High: {max_temp}°F | Low: {min_temp}°F</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class='animate-card'>
                            <h4>{city}</h4>
                            <p>Data unavailable</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # Voice interaction
    if st.button("🔊 Hear Weather"):
        temp = current.get("temperature_2m", "unknown")
        desc = weather_codes.get(current.get("weather_code", 0), {"description": "unknown"})["description"]
        text = f"Current weather in {selected_city}: {temp} degrees Fahrenheit, {desc}."
        text_to_speech(text)

    # Weather trivia
    st.subheader("Weather Trivia")
    st.markdown("🌍 Did you know? The highest recorded temperature was 134°F in Death Valley!")

    # 7-Day Forecast (Displayed at the bottom with stretched horizontal blocks)
    st.subheader("7-Day Forecast")
    for day in forecast_data:
        st.markdown(
            f"""
            <div class='forecast-card' style='width: 100%; padding: 20px;'>
                <b>{day['date']}</b>: {day['condition']}<br>
                High: {day['high']}°C, Low: {day['low']}°C<br>
                Sunrise: {day['sunrise']}, Sunset: {day['sunset']}
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    st.error("Could not fetch weather data for the selected city. Please try again later.")