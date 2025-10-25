# ===============================================================
# DATA PREPARATION SCRIPT - for Solar Potential Model (Simplified)
# ---------------------------------------------------------------
# Cleans raw GHCN data, pivots observations, aggregates by region
# Computes Solar Potential Index (SPI) using only temperature & rain
# Saves clean dataset to /data/processed/
# ===============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

RAW_FILE = "../data/raw/merge_dataset.csv"
OUTPUT_FILE = "../data/processed/solar_dataset.csv"

print("[INFO] Loading raw dataset...")
df = pd.read_csv(RAW_FILE)
print(f"[INFO] Raw rows loaded: {len(df):,}")

# --------------------------------------------------------------
# STEP 1 — Cleaning
# --------------------------------------------------------------
print("[INFO] Cleaning data...")

# Remove invalid rows
df.dropna(subset=["station", "observation", "value", "year", "latitude", "longitude"], inplace=True)
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df["value"].replace([9999, -9999], np.nan, inplace=True)
df.dropna(subset=["value"], inplace=True)

# Filter out unrealistic measurement values
df = df[(df["value"] > -1000) & (df["value"] < 10000)]
df.drop_duplicates(subset=["station", "date", "observation"], inplace=True)

print(f"[INFO] After cleaning: {len(df):,} rows remain")

# --------------------------------------------------------------
# STEP 2 — Pivot observations (long → wide)
# --------------------------------------------------------------
print("[INFO] Pivoting observation types (TAVG, PRCP)...")

pivoted = (
    df.pivot_table(
        index=["station", "year", "latitude", "longitude", "elevation"],
        columns="observation",
        values="value",
        aggfunc="mean"
    ).reset_index()
)

# Convert units (divide by 10 to get °C and mm)
for col in ["TMAX", "TMIN", "TAVG", "PRCP"]:
    if col in pivoted.columns:
        pivoted[col] = pivoted[col] / 10.0

# Rename key features for clarity
pivoted.rename(columns={
    "TAVG": "avg_temp",
    "PRCP": "total_rain"
}, inplace=True)

print(f"[INFO] Pivot complete. Shape: {pivoted.shape}")

# --------------------------------------------------------------
# STEP 3 — Aggregate by region grid (0.1°) + year
# --------------------------------------------------------------
print("[INFO] Aggregating by region (lat/lon rounded to 0.1°) and year...")

pivoted["lat_grid"] = pivoted["latitude"].round(1)
pivoted["lon_grid"] = pivoted["longitude"].round(1)
pivoted["region_id"] = pivoted["lat_grid"].astype(str) + "_" + pivoted["lon_grid"].astype(str)

agg = (
    pivoted.groupby(["region_id", "year"], as_index=False)
    .agg({
        "avg_temp": "mean",
        "total_rain": "sum",
        "elevation": "max",
        "latitude": "mean",
        "longitude": "mean"
    })
)

print(f"[INFO] Aggregated shape: {agg.shape}")

# --------------------------------------------------------------
# STEP 4 — Compute Solar Potential Index (SPI)
# --------------------------------------------------------------
print("[INFO] Computing Solar Potential Index (SPI)...")

# Scale temperature and rainfall (normalize 0–1)
scaler = MinMaxScaler()
agg[["avg_temp_s", "total_rain_s"]] = scaler.fit_transform(
    agg[["avg_temp", "total_rain"]]
)

# SPI Formula (temperature = positive factor, rainfall = negative)
agg["SPI"] = (
    0.6 * agg["avg_temp_s"] - 0.4 * agg["total_rain_s"]
) * 100

# --------------------------------------------------------------
# STEP 5 — Save cleaned dataset
# --------------------------------------------------------------
final = agg[[
    "region_id", "year", "latitude", "longitude", "elevation",
    "avg_temp", "total_rain", "SPI"
]]

final.to_csv(OUTPUT_FILE, index=False)
print(f"[SUCCESS] Cleaned dataset saved → {OUTPUT_FILE}")
print(final.head())
