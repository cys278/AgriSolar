# ===============================================================
# src/api.py — AgriSolar Unified Prediction API (Improved Version)
# ===============================================================
from fastapi import FastAPI, Query
import joblib
import numpy as np

app = FastAPI(
    title="AgriSolar API",
    description="Predict temperature, rainfall, and best crop from coordinates and year.",
    version="2.1.0"
)

# -----------------------------
# Load models
# -----------------------------
crop_model = joblib.load("models/crop_model.joblib")
encoder = joblib.load("models/label_encoder.joblib")
scaler = joblib.load("models/scaler.joblib")
temp_model = joblib.load("models/temperature_model.joblib")
rain_model = joblib.load("models/rainfall_model.joblib")

@app.get("/")
def root():
    return {
        "message": "🌞 AgriSolar API is running",
        "routes": ["/predict_crop", "/predict_crop_by_location"]
    }

# ---------------------------------------------------------------
# Crop prediction from temp & rainfall directly
# ---------------------------------------------------------------
@app.get("/predict_crop")
def predict_crop(temperature: float, rainfall: float):
    # Use average defaults for soil and humidity
    defaults = {"N": 50, "P": 50, "K": 50, "humidity": 70, "ph": 6.5}
    X = np.array([[defaults["N"], defaults["P"], defaults["K"],
                   temperature, defaults["humidity"], defaults["ph"], rainfall]])
    X_scaled = scaler.transform(X)
    pred_encoded = crop_model.predict(X_scaled)[0]
    crop_name = encoder.inverse_transform([pred_encoded])[0]
    return {"temperature": temperature, "rainfall": rainfall, "recommended_crop": crop_name}

# ---------------------------------------------------------------
# Crop prediction from lat/lon/year (auto-predict climate first)
# ---------------------------------------------------------------
@app.get("/predict_crop_by_location")
def predict_crop_by_location(
    latitude: float = Query(..., description="Latitude"),
    longitude: float = Query(..., description="Longitude"),
    year: int = Query(..., description="Year to predict for")
):
    # Predict temperature and rainfall
    X_geo = np.array([[latitude, longitude, year]])
    predicted_temp = float(temp_model.predict(X_geo)[0])
    predicted_rain = float(rain_model.predict(X_geo)[0])

    # Default soil/humidity/pH values (can be adjusted later per region)
    defaults = {"N": 50, "P": 50, "K": 50, "humidity": 70, "ph": 6.5}

    # Combine all inputs for crop model
    X_crop = np.array([[defaults["N"], defaults["P"], defaults["K"],
                        predicted_temp, defaults["humidity"], defaults["ph"], predicted_rain]])
    X_scaled = scaler.transform(X_crop)
    pred_encoded = crop_model.predict(X_scaled)[0]
    crop_name = encoder.inverse_transform([pred_encoded])[0]

    return {
        "latitude": latitude,
        "longitude": longitude,
        "year": year,
        "predicted_temperature": round(predicted_temp, 2),
        "predicted_rainfall": round(predicted_rain, 2),
        "recommended_crop": crop_name
    }
