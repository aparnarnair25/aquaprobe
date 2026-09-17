import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import FADN


# ==========================================================
# SETTINGS
# ==========================================================

FEATURES = ["sst", "sss", "ssh", "wind_u", "wind_v"]

TARGETS = [
    "t_0",
    "t_10",
    "t_30",
    "t_50",
    "t_100",
    "t_200"
]

DEPTHS = [0, 10, 30, 50, 100, 200]

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_FILE = os.path.join(
    BASE_DIR, "data", "processed", "features.csv"
)

TARGET_FILE = os.path.join(
    BASE_DIR, "data", "processed", "targets.csv"
)

CHECKPOINT_DIR = os.path.join(
    BASE_DIR, "model", "checkpoints"
)

CHECKPOINT_FILE = os.path.join(
    CHECKPOINT_DIR, "fadn.pt"
)

NORMALIZATION_FILE = os.path.join(
    CHECKPOINT_DIR, "normalization.json"
)


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ==========================================================
# LOAD DATA
# ==========================================================

print("\nLoading data...")

if not os.path.exists(FEATURE_FILE):
    raise FileNotFoundError(
        f"Features file not found:\n{FEATURE_FILE}\n"
        "Make sure Member 1 has created features.csv."
    )

if not os.path.exists(TARGET_FILE):
    raise FileNotFoundError(
        f"Targets file not found:\n{TARGET_FILE}\n"
        "Make sure Member 1 has created targets.csv."
    )

features_df = pd.read_csv(FEATURE_FILE)
targets_df = pd.read_csv(TARGET_FILE)

print("Features shape:", features_df.shape)
print("Targets shape:", targets_df.shape)


# ==========================================================
# CHECK REQUIRED COLUMNS
# ==========================================================

for column in FEATURES:
    if column not in features_df.columns:
        raise ValueError(
            f"Missing feature column: {column}"
        )

for column in TARGETS:
    if column not in targets_df.columns:
        raise ValueError(
            f"Missing target column: {column}"
        )


# ==========================================================
# SELECT TRAINING ROWS
# ==========================================================

# Use the split column if Member 1 provides it.
if "split" in features_df.columns:

    train_mask = (
        features_df["split"]
        .astype(str)
        .str.lower()
        == "train"
    )

    val_mask = (
        features_df["split"]
        .astype(str)
        .str.lower()
        == "val"
    )

    # If validation is called "validation", support that too.
    if val_mask.sum() == 0:
        val_mask = (
            features_df["split"]
            .astype(str)
            .str.lower()
            == "validation"
        )

    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise ValueError(
            "The split column must contain 'train' and 'val' values."
        )

    X_train = features_df.loc[train_mask, FEATURES].values
    X_val = features_df.loc[val_mask, FEATURES].values

    # Targets are joined using the same row indices.
    y_train = targets_df.loc[train_mask, TARGETS].values
    y_val = targets_df.loc[val_mask, TARGETS].values

else:
    # Fallback: create an 80/20 split.
    print("No split column found. Creating an 80/20 split.")

    X = features_df[FEATURES].values
    y = targets_df[TARGETS].values

    rng = np.random.default_rng(42)

    indices = np.arange(len(X))
    rng.shuffle(indices)

    split_index = int(0.8 * len(indices))

    train_indices = indices[:split_index]
    val_indices = indices[split_index:]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_val = X[val_indices]
    y_val = y[val_indices]


print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))


# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

X_train = np.asarray(X_train, dtype=np.float32)
X_val = np.asarray(X_val, dtype=np.float32)

y_train = np.asarray(y_train, dtype=np.float32)
y_val = np.asarray(y_val, dtype=np.float32)

if np.isnan(X_train).any() or np.isnan(X_val).any():
    raise ValueError("Input features contain missing values.")

if np.isnan(y_train).any() or np.isnan(y_val).any():
    raise ValueError("Target values contain missing values.")


# ==========================================================
# STANDARDIZE INPUT FEATURES
# ==========================================================

# IMPORTANT:
# Mean and standard deviation are calculated ONLY from
# the training data.

feature_mean = X_train.mean(axis=0)
feature_std = X_train.std(axis=0)

# Prevent division by zero.
feature_std[feature_std < 1e-8] = 1.0

X_train_scaled = (
    X_train - feature_mean
) / feature_std

X_val_scaled = (
    X_val - feature_mean
) / feature_std


# ==========================================================
# CONVERT TO PYTORCH TENSORS
# ==========================================================

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
)

X_val_tensor = torch.tensor(
    X_val_scaled,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val,
    dtype=torch.float32
)


# ==========================================================
# DATA LOADER
# ==========================================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ==========================================================
# CREATE MODEL
# ==========================================================

model = FADN(
    input_size=5,
    hidden_size=32,
    output_size=6
).to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================================
# TRAINING
# ==========================================================

print("\nStarting training...")
print("----------------------------------------")

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0.0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        predictions, attention_weights = model(batch_X)

        loss = criterion(
            predictions,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_train_loss += (
            loss.item() * batch_X.size(0)
        )

    average_train_loss = (
        total_train_loss / len(train_dataset)
    )

    # ------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------

    model.eval()

    with torch.no_grad():

        val_predictions, _ = model(
            X_val_tensor.to(device)
        )

        val_loss = criterion(
            val_predictions,
            y_val_tensor.to(device)
        ).item()

    # ------------------------------------------------------
    # SAVE BEST MODEL
    # ------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        os.makedirs(
            CHECKPOINT_DIR,
            exist_ok=True
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_size": 5,
                "hidden_size": 32,
                "output_size": 6,
                "depths": DEPTHS
            },
            CHECKPOINT_FILE
        )

    # Print progress
    if (
        (epoch + 1) % 10 == 0
        or epoch == 0
    ):
        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS} "
            f"| Train Loss: {average_train_loss:.6f} "
            f"| Val Loss: {val_loss:.6f}"
        )


# ==========================================================
# SAVE NORMALIZATION STATISTICS
# ==========================================================

normalization_data = {
    "features": FEATURES,
    "mean": feature_mean.tolist(),
    "std": feature_std.tolist()
}

with open(
    NORMALIZATION_FILE,
    "w"
) as f:
    json.dump(
        normalization_data,
        f,
        indent=4
    )


# ==========================================================
# FINAL MESSAGE
# ==========================================================

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print("\nBest validation loss:", best_val_loss)

print("\nModel saved to:")
print(CHECKPOINT_FILE)

print("\nNormalization statistics saved to:")
print(NORMALIZATION_FILE)

print("\nDepths:")
print(DEPTHS)

print("\nNext step:")
print("Create predict.py")