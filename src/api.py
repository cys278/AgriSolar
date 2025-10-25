# ===============================================================
# src/api.py — AgriSolar Unified Prediction API (Region-Aware Version)
# ===============================================================

from fastapi import FastAPI, Query
import joblib
import numpy as np

# ---------------------------------------------------------------
# Initialize app
# ---------------------------------------------------------------
app = FastAPI(
    title="AgriSolar API",
    description="Predicts temperature, rainfall, and best crop from coordinates and year — with region-aware defaults.",
    version="3.0.0"
)

# ---------------------------------------------------------------
# Load trained models
# ---------------------------------------------------------------
crop_model = joblib.load("ML_MODELS/crop_model.joblib")
encoder = joblib.load("ML_MODELS/label_encoder.joblib")
scaler = joblib.load("ML_MODELS/scaler.joblib")
temp_model = joblib.load("ML_MODELS/temperature_model.joblib")
rain_model = joblib.load("ML_MODELS/rainfall_model.joblib")
solar_model = joblib.load("ML_MODELS/solar_model.joblib") 


@app.get("/")
def root():
    return {
        "message": "🌞 AgriSolar API is running successfully!",
        "routes": ["/predict_crop", "/predict_crop_by_location"]
    }


# ---------------------------------------------------------------
# Direct crop prediction (temperature & rainfall input)
# ---------------------------------------------------------------
@app.get("/predict_crop")
def predict_crop(
    temperature: float = Query(..., description="Average temperature (°C)"),
    rainfall: float = Query(..., description="Total rainfall (mm)")
):
    """Predict crop directly from temperature & rainfall."""
    defaults = {"N": 60, "P": 60, "K": 60, "humidity": 70, "ph": 6.5}

    X = np.array([[defaults["N"], defaults["P"], defaults["K"],
                   temperature, defaults["humidity"], defaults["ph"], rainfall]])
    X_scaled = scaler.transform(X)
    pred_encoded = crop_model.predict(X_scaled)[0]
    crop_name = encoder.inverse_transform([pred_encoded])[0]

    return {
        "temperature": temperature,
        "rainfall": rainfall,
        "recommended_crop": crop_name
    }


# ---------------------------------------------------------------
# Region-aware prediction (latitude, longitude, year)
# ---------------------------------------------------------------
@app.get("/predict_crop_by_location")
def predict_crop_by_location(
    latitude: float = Query(..., description="Latitude of location"),
    longitude: float = Query(..., description="Longitude of location"),
    year: int = Query(..., description="Target year")
):
    """Predicts temperature & rainfall using coordinates + year,
    applies region-aware soil defaults, and recommends crop."""

    # Step 1. Predict climate from geolocation & year
    X_geo = np.array([[latitude, longitude, year]])
    predicted_temp = float(temp_model.predict(X_geo)[0])
    predicted_rain = float(rain_model.predict(X_geo)[0])

    # Step 2. Region-aware soil & humidity defaults
    if abs(latitude) < 10:  # Equatorial — hot, humid, acidic
        defaults = {"N": 40, "P": 40, "K": 40, "humidity": 85, "ph": 5.8}
    elif 10 <= abs(latitude) < 25:  # Tropical/subtropical
        defaults = {"N": 60, "P": 50, "K": 45, "humidity": 75, "ph": 6.3}
    elif 25 <= abs(latitude) < 40:  # Temperate
        defaults = {"N": 70, "P": 60, "K": 60, "humidity": 65, "ph": 6.8}
    elif 40 <= abs(latitude) < 55:  # Cooler
        defaults = {"N": 80, "P": 70, "K": 65, "humidity": 55, "ph": 7.0}
    else:  # Polar / very cold
        defaults = {"N": 50, "P": 60, "K": 55, "humidity": 50, "ph": 6.5}

    # Step 3. Predict crop
    X_crop = np.array([[defaults["N"], defaults["P"], defaults["K"],
                        predicted_temp, defaults["humidity"], defaults["ph"], predicted_rain]])
    X_scaled = scaler.transform(X_crop)
    pred_encoded = crop_model.predict(X_scaled)[0]
    crop_name = encoder.inverse_transform([pred_encoded])[0]

    # Step 4. Predict Solar Potential (SPI)
    X_solar = np.array([[latitude, longitude, year, predicted_temp, predicted_rain]])
    predicted_spi = float(solar_model.predict(X_solar)[0])

    # Step 4. Return JSON response
    return {
        "latitude": latitude,
        "longitude": longitude,
        "year": year,
        "predicted_temperature": round(predicted_temp, 2),
        "predicted_rainfall": round(predicted_rain, 2),
        "region_defaults": defaults,
        "solar_potential_index": round(predicted_spi, 2),
        "recommended_crop": crop_name
    }
