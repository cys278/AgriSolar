# ===============================================================
# SPLIT DATASET INTO TRAINING AND TEST SETS
# ---------------------------------------------------------------
# Input : solar_training_dataset.csv
# Output: training_data.csv (2/3 of rows)
#         testing_data.csv  (1/3 of rows)
# ===============================================================

import pandas as pd
from sklearn.model_selection import train_test_split

# 1️⃣ Load your processed dataset
INPUT_FILE = "../data/processed/solar_dataset.csv"
TRAIN_FILE = "../data/processed/solar_data_training.csv"
TEST_FILE = "../data/processed/testing_data_solar.csv"

print("[INFO] Loading dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"[INFO] Total rows in dataset: {len(df):,}")

# 2️⃣ Split data into 2/3 (train) and 1/3 (test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"[INFO] Training data size: {len(train_df):,}")
print(f"[INFO] Testing data size:  {len(test_df):,}")

# 3️⃣ Save both subsets to separate CSV files
train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

print(f"[SUCCESS] Files saved:")
print(f"   → {TRAIN_FILE}")
print(f"   → {TEST_FILE}")

# Optional: Show sample preview
print("\n[INFO] Preview of training data:")
print(train_df.head())
