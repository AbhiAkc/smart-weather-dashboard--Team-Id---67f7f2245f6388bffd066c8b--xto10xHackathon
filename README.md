# 🌍 Smart Weather Forecast Dashboard

An interactive and intelligent weather forecasting dashboard built with **Streamlit**, utilizing real-time data and machine learning to provide insightful weather predictions with a modern, user-friendly interface.

---

## 📘 Project Overview

This project is a **smart weather forecast system** that combines historical data analysis with live updates to generate accurate predictions. It empowers users to:

- View current weather across cities globally
- Compare forecasts between multiple cities
- Understand predicted patterns using machine learning

💡 **Why it matters**: With extreme weather events increasing globally, accurate and interpretable forecasts are more important than ever. This app helps make weather insights accessible to everyone—from casual users to data enthusiasts.

---

## 🔧 Key Features & Technologies

### 🛠 Technologies Used

- **Frontend**: [Streamlit](https://streamlit.io/)
- **ML Models**: Linear Regression, LSTM (via TensorFlow/Keras)
- **Geolocation**: Geopy & Folium for mapping
- **APIs**: Open-Meteo, OpenWeatherMap
- **Database**: Supabase
- **Others**: gTTS (Text-to-Speech), Plotly, Requests-Cache, Pandas

### ✨ Features

- 🔍 Search for any city with automatic geocoding
- 🧠 Choose between **Linear Regression** or **LSTM** for prediction
- 📊 7-day weather forecasting with model explainability
- 📍 Interactive weather map with live location pins
- ⚠️ Dynamic weather alerts for extreme conditions
- 🌤 Sunrise/Sunset timing and day planning suggestions
- 🔈 Optional **Text-to-Speech** for weather summaries
- 📈 Multi-city comparison cards
- 🧾 Prediction storage using **Supabase**

---

## ⚙️ Setup Instructions

### 🔄 Quick Start

```bash
git clone https://github.com/your-username/smart-weather-dashboard.git
cd smart-weather-dashboard
pip install -r requirements.txt
streamlit run app.py

Data Handling & Model Explainability
📥 Data Flow
Input: Real-time weather and 30-day historical data via Open-Meteo API

Preprocessing:

Missing values handled via interpolation

Normalization for LSTM model

Model Selection:

Linear Regression: Uses day_of_year for seasonal trend fitting

LSTM: Sequence learning using last 7 days of temperature

🧠 Model Explainability
Linear Regression: Easy to interpret slope/seasonal trends

LSTM: Black-box model, but predictions are validated against recent trends

Both models store their outputs in Supabase, enabling traceability and later analysis.

🙌 Credits
Created by Abhishek Kumar Chaudhary and Abhishek Kumar
for the xto10x Hackathon event organized by Masai School.