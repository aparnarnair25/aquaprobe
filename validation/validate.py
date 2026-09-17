import os
import sys
import numpy as np
import pandas as pd

# Allow importing the prediction module
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "model")

if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from predict import predict


FEATURES = [
    "sst",
    "sss",
    "ssh",
    "wind_u",
    "wind_v"
]

TEMPERATURE_COLUMNS = [
    "t_0",
    "t_10",
    "t_30",
    "t_50",
    "t_100",
    "t_200"
]

DATA_FILE = os.path.join(
    MODEL_DIR,
    "data",
    "raw",
    "aquaprobe_raw.csv"
)


def calculate_validation_metrics(actual, predicted):
    """
    Calculate RMSE, MAE, correlation and bias.
    """

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError(
            "Actual and predicted arrays must have "
            "the same shape."
        )

    rmse = np.sqrt(
        np.mean((predicted - actual) ** 2)
    )

    mae = np.mean(
        np.abs(predicted - actual)
    )

    bias = np.mean(
        predicted - actual
    )

    actual_flat = actual.flatten()
    predicted_flat = predicted.flatten()

    if (
        len(actual_flat) < 2
        or np.std(actual_flat) == 0
        or np.std(predicted_flat) == 0
    ):
        correlation = 0.0
    else:
        correlation = np.corrcoef(
            actual_flat,
            predicted_flat
        )[0, 1]

    return {
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "correlation": round(float(correlation), 4),
        "bias": round(float(bias), 4)
    }


def run_validation(split="test"):
    """
    Run FADN prediction on the selected dataset split
    and compare predictions with ground-truth temperatures.
    """

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = (
        FEATURES
        + TEMPERATURE_COLUMNS
        + ["split"]
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df[
        df["split"].str.lower() == split.lower()
    ].copy()

    if df.empty:
        raise ValueError(
            f"No rows found for split: {split}"
        )

    actual_values = []
    predicted_values = []

    print(f"Running validation on '{split}' split...")
    print(f"Number of samples: {len(df)}")

    for _, row in df.iterrows():

        features = {
            feature: float(row[feature])
            for feature in FEATURES
        }

        result = predict(features)

        predicted = result["temperature"]

        actual = [
            float(row[column])
            for column in TEMPERATURE_COLUMNS
        ]

        if len(predicted) != len(actual):
            raise ValueError(
                "Prediction and ground-truth temperature "
                "profiles have different lengths."
            )

        predicted_values.append(predicted)
        actual_values.append(actual)

    metrics = calculate_validation_metrics(
        actual_values,
        predicted_values
    )

    return metrics


if __name__ == "__main__":

    try:

        metrics = run_validation("test")

        print("\nValidation Results")
        print("------------------")
        print(f"RMSE        : {metrics['rmse']}")
        print(f"MAE         : {metrics['mae']}")
        print(f"Correlation : {metrics['correlation']}")
        print(f"Bias        : {metrics['bias']}")

    except Exception as error:

        print("\nValidation failed.")
        print("Reason:")
        print(error)