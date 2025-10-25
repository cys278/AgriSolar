# ===============================================================
# src/train_climate_models.py — Train temperature & rainfall models
# ===============================================================
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

INPUT_FILE = "data/processed/agriculture_clean.csv"
os.makedirs("models", exist_ok=True)

print("[INFO] Loading cleaned agriculture data...")
df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=["latitude", "longitude", "year", "TAVG", "PRCP"])

# Prepare features
X = df[["latitude", "longitude", "year"]]
y_temp = df["TAVG"]
y_rain = df["PRCP"]

print("[INFO] Training RandomForestRegressor for temperature...")
temp_model = RandomForestRegressor(n_estimators=300, random_state=42)
temp_model.fit(X, y_temp)

print("[INFO] Training RandomForestRegressor for rainfall...")
rain_model = RandomForestRegressor(n_estimators=300, random_state=42)
rain_model.fit(X, y_rain)

# Save models
joblib.dump(temp_model, "models/temperature_model.joblib")
joblib.dump(rain_model, "models/rainfall_model.joblib")

print("✅ Saved temperature_model.joblib and rainfall_model.joblib")
