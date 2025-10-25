# ===============================================================
# MODEL TRAINING SCRIPT - for Solar Potential Model
# ---------------------------------------------------------------
# Loads processed dataset from /data/processed/
# Cleans missing/invalid values
# Tracks rows lost during cleaning
# Trains XGBRegressor to predict SPI
# Saves trained model to /models/
# ===============================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import joblib

# --------------------------------------------------------------
# STEP 0 — File paths
# --------------------------------------------------------------
INPUT_FILE = "../data/processed/solar_data_training.csv"
MODEL_FILE = "../models/solar_model.joblib"

# Ensure the model folder exists
os.makedirs("../models", exist_ok=True)

print("[INFO] Loading processed dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"[INFO] Dataset loaded. Shape: {df.shape}")  # BEFORE cleaning row count

# --------------------------------------------------------------
# STEP 1 — Data integrity checks / cleaning
# --------------------------------------------------------------
print("[INFO] Checking for invalid or missing values in dataset...")

# Store the initial row count
initial_rows = len(df)

# Replace infinities with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Show missing value summary
print("[DEBUG] Missing values per column BEFORE cleaning:")
print(df.isna().sum())

# Drop rows where SPI or key weather variables are missing
required_cols = ["SPI", "avg_temp", "total_rain", "total_snow",
                 "latitude", "longitude", "year"]
df = df.dropna(subset=required_cols)

# Handle avg_wind (fill with mean if missing)
if "avg_wind" in df.columns:
    if df["avg_wind"].isna().sum() > 0:
        df["avg_wind"] = df["avg_wind"].fillna(df["avg_wind"].mean())
        print("[INFO] Filled missing avg_wind values with mean.")
else:
    df["avg_wind"] = 0.0
    print("[WARN] avg_wind column missing — created constant 0.0")

# Track cleaning results
cleaned_rows = len(df)
dropped_rows = initial_rows - cleaned_rows
percent_retained = (cleaned_rows / initial_rows) * 100

print(f"[INFO] Rows before cleaning:  {initial_rows:,}")
print(f"[INFO] Rows after cleaning:   {cleaned_rows:,}")
print(f"[INFO] Rows dropped:          {dropped_rows:,} ({100 - percent_retained:.2f}% removed)")
print(f"[INFO] Data retained:         {percent_retained:.2f}%")

# --------------------------------------------------------------
# STEP 2 — Prepare features & target
# --------------------------------------------------------------
print("[INFO] Preparing features and target for training...")

feature_cols = [
    "latitude", "longitude", "elevation", "year",
    "avg_temp", "total_rain", "total_snow", "avg_wind"
]
X = df[feature_cols]
y = df["SPI"]

# Safety check for any remaining NaN
if X.isna().sum().sum() > 0 or y.isna().sum() > 0:
    raise ValueError("[ERROR] NaN values still present after cleaning. Please recheck data!")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("[INFO] Data split complete:")
print(f"    Train size: {X_train.shape[0]}")
print(f"    Test size:  {X_test.shape[0]}")

# --------------------------------------------------------------
# STEP 3 — Train the model
# --------------------------------------------------------------
print("[INFO] Training XGBoost Regressor for Solar Potential...")

solar_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

solar_model.fit(X_train, y_train)
print("[SUCCESS] Model training complete.")

# --------------------------------------------------------------
# STEP 4 — Evaluate performance
# --------------------------------------------------------------
y_pred = solar_model.predict(X_test)

# Calculate R² and RMSE (manual sqrt for compatibility)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"[METRICS] R² score: {r2:.3f}")
print(f"[METRICS] RMSE:     {rmse:.3f}")

# --------------------------------------------------------------
# STEP 5 — Save trained model
# --------------------------------------------------------------
joblib.dump(solar_model, MODEL_FILE)
print(f"[SUCCESS] Model saved → {MODEL_FILE}")

# --------------------------------------------------------------
# STEP 6 — Quick test prediction
# --------------------------------------------------------------
print("[TEST] Example prediction using first test sample:")
sample = X_test.iloc[0]
pred = solar_model.predict(sample.values.reshape(1, -1))[0]
print(f"    Location ({sample['latitude']}, {sample['longitude']}) "
      f"year {sample['year']} → predicted SPI = {pred:.2f}")
