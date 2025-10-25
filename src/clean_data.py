"""
clean_data.py — GHCN Data Cleaning Script
Usage:
    python src/clean_data.py
"""

import os
import pandas as pd
import numpy as np
import duckdb

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_FILE = f"{PROCESSED_DIR}/agriculture_clean.csv"

def clean_and_merge():
    # Collect all sample files
    csv_files = [f for f in os.listdir(RAW_DIR) if f.startswith("ghcn_sample_") and f.endswith(".csv")]
    print(f"Found {len(csv_files)} partial files...")

    # Use DuckDB for efficient merging
    con = duckdb.connect(database=':memory:')
    query = f"""
        SELECT 
            station,
            date,
            observation,
            value,
            year,
            latitude,
            longitude,
            elevation,
            name
        FROM read_csv_auto({[os.path.join(RAW_DIR, f) for f in csv_files]}, SAMPLE_SIZE=-1)
        WHERE value NOT IN (9999,99999) AND value IS NOT NULL
    """
    df = con.execute(query).fetchdf()
    print("Loaded and filtered raw data:", df.shape)

    # Pivot to wide format
    df_pivot = df.pivot_table(index=["station","year","latitude","longitude","elevation","name"],
                              columns="observation",
                              values="value",
                              aggfunc="mean").reset_index()

    # Convert units (divide by 10 where needed)
    for col in ["TMAX","TMIN","TAVG","AWND","PRCP","SNOW","SNWD"]:
        if col in df_pivot.columns:
            df_pivot[col] = df_pivot[col].apply(lambda x: x/10 if x and abs(x) > 50 else x)

    # Remove invalid coordinates and outliers
    df_pivot = df_pivot[
        (df_pivot["latitude"].between(-90, 90)) &
        (df_pivot["longitude"].between(-180, 180)) &
        (df_pivot["TMAX"] < 60) &
        (df_pivot["TMIN"] > -60)
    ]

    # Save cleaned dataset
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_pivot.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Cleaned dataset saved to {OUTPUT_FILE}, shape={df_pivot.shape}")

if __name__ == "__main__":
    clean_and_merge()
