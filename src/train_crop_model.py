# ===============================================================
# src/train_crop_model.py — Improved Crop Recommendation Model
# ===============================================================
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import joblib

# -----------------------------
# Paths
# -----------------------------
CROP_DATA = "data/external/Crop_recommendation.csv"
MODEL_OUT = "models/crop_model.joblib"
ENCODER_OUT = "models/label_encoder.joblib"
SCALER_OUT = "models/scaler.joblib"

os.makedirs("models", exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
print("[INFO] Loading crop dataset...")
df = pd.read_csv(CROP_DATA)

# Keep full set of features
df = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]].dropna()

# Encode labels
encoder = LabelEncoder()
df["label_encoded"] = encoder.fit_transform(df["label"])
joblib.dump(encoder, ENCODER_OUT)
print(f"[INFO] Encoded crops: {list(encoder.classes_)}")

# Prepare data
X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = df["label_encoded"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_OUT)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# -----------------------------
# Train model (tuned XGBoost)
# -----------------------------
print("[INFO] Training XGBoost classifier...")
model = XGBClassifier(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.2,
    reg_lambda=1.5,
    random_state=42,
    tree_method="hist"
)
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"[METRIC] Accuracy: {acc:.3f}")
print(f"[METRIC] F1-score: {f1:.3f}")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# -----------------------------
# Save model + artifacts
# -----------------------------
joblib.dump(model, MODEL_OUT)
print(f"✅ Model saved to {MODEL_OUT}")
print(f"✅ Scaler saved to {SCALER_OUT}")
print(f"✅ Label encoder saved to {ENCODER_OUT}")
