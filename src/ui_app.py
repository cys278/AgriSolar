# ===============================================================
# src/ui_app.py — Simple UI to test AgriSolar API
# ===============================================================
import streamlit as st
import requests

st.set_page_config(page_title="AgriSolar Crop Predictor 🌾", layout="centered")
st.title("🌞 AgriSolar — Smart Crop Recommendation")
st.write("Enter a location (latitude, longitude) and a target year to predict climate and crop suitability.")

lat = st.number_input("Latitude", value=25.25, format="%.4f")
lon = st.number_input("Longitude", value=55.36, format="%.4f")
year = st.number_input("Year", value=2026, step=1)

if st.button("Predict Crop"):
    url = f"http://127.0.0.1:8000/predict_crop_by_location?latitude={lat}&longitude={lon}&year={year}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            st.success(f"Recommended Crop: **{data['recommended_crop'].title()}**")
            st.metric("Predicted Temperature (°C)", f"{data['predicted_temperature']} °C")
            st.metric("Predicted Rainfall (mm)", f"{data['predicted_rainfall']} mm")
        else:
            st.error(f"Error: {response.status_code}")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
