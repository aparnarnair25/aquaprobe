import os
import pandas as pd

# Paths
BASE_DIR = BASE_DIR = os.path.dirname(os.path.dirname(__file__))

RAW_FILE = os.path.join(
    BASE_DIR, "model", "data", "raw", "aquaprobe_raw.csv"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR, "data", "processed"
)

FEATURE_FILE = os.path.join(
    PROCESSED_DIR, "features.csv"
)

TARGET_FILE = os.path.join(
    PROCESSED_DIR, "targets.csv"
)

# Create processed folder if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load raw data
print("Loading raw data...")

df = pd.read_csv(RAW_FILE)

print("Raw data shape:", df.shape)

# Features
FEATURES = [
    "sst",
    "sss",
    "ssh",
    "wind_u",
    "wind_v"
]

# Targets
TARGETS = [
    "t_0",
    "t_10",
    "t_30",
    "t_50",
    "t_100",
    "t_200"
]

# Check columns
for column in FEATURES + TARGETS:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

# Create feature and target datasets
features_df = df[FEATURES].copy()
targets_df = df[TARGETS].copy()

# Save
features_df.to_csv(FEATURE_FILE, index=False)
targets_df.to_csv(TARGET_FILE, index=False)

print("Features shape:", features_df.shape)
print("Targets shape:", targets_df.shape)

print("\nCreated:")
print(FEATURE_FILE)
print(TARGET_FILE)