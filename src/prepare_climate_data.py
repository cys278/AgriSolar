# ===============================================================
# src/prepare_climate_data.py — Prepare standardized climate file
# ===============================================================
import os
import pandas as pd

INPUT_FILE = "data/processed/agriculture_clean.csv"
OUTPUT_FILE = "data/processed/climate_for_crop_model.csv"
os.makedirs("data/processed", exist_ok=True)

print("[INFO] Loading cleaned climate data...")
df = pd.read_csv(INPUT_FILE)

df_out = pd.DataFrame({
    "station": df["station"],
    "year": df["year"],
    "latitude": df["latitude"],
    "longitude": df["longitude"],
    "elevation": df["elevation"],
    "name": df["name"],
    "temperature": df["TAVG"],
    "rainfall": df["PRCP"]
})

df_out = df_out.dropna(subset=["temperature", "rainfall"])
df_out.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved standardized climate data to {OUTPUT_FILE}")
