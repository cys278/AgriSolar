# ===============================================================
# MODEL TRAINING SCRIPT - for Solar Potential Model
# ---------------------------------------------------------------
# Loads processed dataset from /data/processed/
# Trains XGBRegressor to predict SPI
# Saves model to /models/
# ===============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import joblib

INPUT_FILE = "../data/processed/solar_training_dataset.csv"
MODEL_FILE = "../models/solar_model.joblib"

print("[INFO] Loading processed dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"[INFO] Dataset loaded. Shape: {df.shape}")

# --------------------------------------------------------------
# STEP 1 — Prepare features & target
# --------------------------------------------------------------
X = df[["latitude", "longitude", "elevation", "year",
        "avg_temp", "total_rain", "total_snow", "avg_wind"]]
y = df["SPI"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("[INFO] Data split complete:")
print(f"    Train size: {X_train.shape[0]}")
print(f"    Test size:  {X_test.shape[0]}")

# --------------------------------------------------------------
# STEP 2 — Train the model
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
# STEP 3 — Evaluate performance
# --------------------------------------------------------------
y_pred = solar_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"[METRICS] R² score: {r2:.3f}")
print(f"[METRICS] RMSE:     {rmse:.3f}")

# --------------------------------------------------------------
# STEP 4 — Save trained model
# --------------------------------------------------------------
joblib.dump(solar_model, MODEL_FILE)
print(f"[SUCCESS] Model saved → {MODEL_FILE}")

# --------------------------------------------------------------
# STEP 5 — Optional: Quick test prediction
# --------------------------------------------------------------
print("[TEST] Example prediction using first test sample:")
sample = X_test.iloc[0]
pred = solar_model.predict(sample.values.reshape(1, -1))[0]
print(f"    Location ({sample['latitude']}, {sample['longitude']}) year {sample['year']} → predicted SPI = {pred:.2f}")
