import os
import json
import torch

from model import FADN


# ==========================================================
# SETTINGS
# ==========================================================

FEATURES = [
    "sst",
    "sss",
    "ssh",
    "wind_u",
    "wind_v"
]

DEPTHS = [0, 10, 30, 50, 100, 200]

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CHECKPOINT_FILE = os.path.join(
    BASE_DIR,
    "model",
    "checkpoints",
    "fadn.pt"
)

NORMALIZATION_FILE = os.path.join(
    BASE_DIR,
    "model",
    "checkpoints",
    "normalization.json"
)


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model():
    """
    Load the trained FADN model.
    """

    if not os.path.exists(CHECKPOINT_FILE):
        raise FileNotFoundError(
            "Trained model not found.\n"
            f"Expected location:\n{CHECKPOINT_FILE}\n\n"
            "Run train.py first."
        )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=device
    )

    model = FADN(
        input_size=checkpoint.get("input_size", 5),
        hidden_size=checkpoint.get("hidden_size", 32),
        output_size=checkpoint.get("output_size", 6)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    return model


# ==========================================================
# LOAD NORMALIZATION
# ==========================================================

def load_normalization():
    """
    Load the training-data mean and standard deviation.
    """

    if not os.path.exists(NORMALIZATION_FILE):
        raise FileNotFoundError(
            "Normalization file not found.\n"
            f"Expected location:\n{NORMALIZATION_FILE}\n\n"
            "Run train.py first."
        )

    with open(
        NORMALIZATION_FILE,
        "r"
    ) as f:

        normalization = json.load(f)

    return (
        normalization["mean"],
        normalization["std"]
    )


# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict(features: dict):
    """
    Predict subsurface temperature.

    Input example:

    {
        "sst": 28.5,
        "sss": 35.2,
        "ssh": 0.35,
        "wind_u": 4.0,
        "wind_v": -2.0
    }

    Returns:

    {
        "depths": [...],
        "temperature": [...],
        "attention_weights": [...]
    }
    """

    # ------------------------------------------------------
    # Check input
    # ------------------------------------------------------

    for feature in FEATURES:

        if feature not in features:
            raise ValueError(
                f"Missing feature: {feature}"
            )

    # ------------------------------------------------------
    # Load model and normalization
    # ------------------------------------------------------

    model = load_model()

    mean, std = load_normalization()

    # ------------------------------------------------------
    # Create input in correct order
    # ------------------------------------------------------

    input_values = [
        float(features["sst"]),
        float(features["sss"]),
        float(features["ssh"]),
        float(features["wind_u"]),
        float(features["wind_v"])
    ]

    # ------------------------------------------------------
    # Standardize using TRAINING statistics
    # ------------------------------------------------------

    normalized_values = [
        (value - m) / s
        for value, m, s
        in zip(input_values, mean, std)
    ]

    # ------------------------------------------------------
    # Convert to tensor
    # ------------------------------------------------------

    x = torch.tensor(
        [normalized_values],
        dtype=torch.float32
    ).to(device)

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    with torch.no_grad():

        temperature, attention = model(x)

    # ------------------------------------------------------
    # Convert tensors to Python lists
    # ------------------------------------------------------

    temperature = (
        temperature
        .cpu()
        .numpy()[0]
        .tolist()
    )

    attention = (
        attention
        .cpu()
        .numpy()[0]
        .tolist()
    )

    # ------------------------------------------------------
    # Return stable API format
    # ------------------------------------------------------

    result = {
        "depths": DEPTHS,
        "temperature": temperature,
        "attention_weights": attention
    }

    return result


# ==========================================================
# SIMPLE TEST
# ==========================================================

if __name__ == "__main__":

    print("Testing prediction pipeline...")

    sample_features = {
        "sst": 28.5,
        "sss": 35.2,
        "ssh": 0.35,
        "wind_u": 4.0,
        "wind_v": -2.0
    }

    try:

        result = predict(
            sample_features
        )

        print("\nPrediction successful!")

        print("\nDepths:")
        print(result["depths"])

        print("\nTemperature:")
        print(result["temperature"])

        print("\nAttention weights:")
        print(result["attention_weights"])

    except Exception as e:

        print("\nPrediction could not be completed.")
        print("Reason:")
        print(e)