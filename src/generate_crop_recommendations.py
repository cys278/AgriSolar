# ===============================================================
# src/generate_crop_recommendations.py — Predict crops for all stations
# ===============================================================
import os
import pandas as pd
import numpy as np
import joblib

CLIMATE_FILE = "data/processed/climate_for_crop_model.csv"
OUTPUT_FILE = "data/processed/crop_recommendations.csv"

print("[INFO] Loading climate dataset...")
climate_df = pd.read_csv(CLIMATE_FILE)

print("[INFO] Loading models...")
temp_model = joblib.load("models/temperature_model.joblib")
rain_model = joblib.load("models/rainfall_model.joblib")
crop_model = joblib.load("models/crop_model.joblib")
encoder = joblib.load("models/label_encoder.joblib")

# Predict temp/rain if missing
if "temperature" not in climate_df or climate_df["temperature"].isna().any():
    climate_df["temperature"] = temp_model.predict(climate_df[["latitude", "longitude", "year"]])
if "rainfall" not in climate_df or climate_df["rainfall"].isna().any():
    climate_df["rainfall"] = rain_model.predict(climate_df[["latitude", "longitude", "year"]])

print("[INFO] Predicting crops...")
X_pred = climate_df[["temperature", "rainfall"]]
climate_df["recommended_crop"] = encoder.inverse_transform(crop_model.predict(X_pred))

cols = ["station", "year", "latitude", "longitude", "elevation", "name",
        "temperature", "rainfall", "recommended_crop"]
climate_df[cols].to_csv(OUTPUT_FILE, index=False)
print(f"[SUCCESS] Saved to {OUTPUT_FILE}")
