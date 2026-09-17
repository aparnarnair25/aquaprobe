import os
import sys
import numpy as np

# Allow importing files from the model folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

sys.path.insert(0, MODEL_DIR)

from predict import predict


FEATURE_NAMES = [
    "sst",
    "sss",
    "ssh",
    "wind_u",
    "wind_v"
]

DEPTHS = [0, 10, 30, 50, 100, 200]
def get_feature_importance(features):
    """
    Get model attention weights for the five input features.
    """

    result = predict(features)

    attention = result["attention_weights"]

    importance = {}

    for name, weight in zip(FEATURE_NAMES, attention):
        importance[name] = float(weight)

    return importance
def generate_scientific_insight(depth, temperatures, lower_bound=None, upper_bound=None):
    """
    Generate a simple descriptive insight from the predicted
    temperature profile.
    """

    temperatures = np.asarray(temperatures, dtype=float)

    # Temperature change from surface to the selected depth
    surface_temperature = temperatures[0]

    # Find the requested depth
    if depth in DEPTHS:
        index = DEPTHS.index(depth)
        selected_temperature = temperatures[index]
    else:
        index = int(np.argmin(np.abs(np.array(DEPTHS) - depth)))
        selected_temperature = temperatures[index]

    temperature_drop = surface_temperature - selected_temperature

    # Basic profile description
    if temperature_drop >= 3.0:
        insight = (
            f"Temperature decreases substantially from the surface to "
            f"{DEPTHS[index]} m, indicating strong thermal stratification "
            f"and a possible thermocline."
        )

    elif temperature_drop >= 1.0:
        insight = (
            f"Temperature decreases moderately with depth to "
            f"{DEPTHS[index]} m, indicating moderate stratification."
        )

    else:
        insight = (
            f"Temperature remains relatively similar with depth to "
            f"{DEPTHS[index]} m, indicating weak stratification."
        )

    # Add uncertainty description if bounds are available
    if lower_bound is not None and upper_bound is not None:
        lower_bound = np.asarray(lower_bound, dtype=float)
        upper_bound = np.asarray(upper_bound, dtype=float)

        uncertainty_width = upper_bound[index] - lower_bound[index]

        if uncertainty_width > 2.0:
            insight += " The prediction interval is relatively wide, so uncertainty is higher."
        else:
            insight += " The prediction interval is relatively narrow."

    return insight
def explain(features, depth=50):
    """
    Return feature importance and a scientific insight.
    """

    result = predict(features)

    importance = get_feature_importance(features)

    insight = generate_scientific_insight(
        depth=depth,
        temperatures=result["temperature"],
        lower_bound=result.get("lower_bound"),
        upper_bound=result.get("upper_bound")
    )

    return {
        "depth": depth,
        "importance": importance,
        "insight": insight
    }
if __name__ == "__main__":

    test_features = {
        "sst": 28.5,
        "sss": 35.0,
        "ssh": 0.2,
        "wind_u": 3.0,
        "wind_v": -2.0
    }

    result = explain(test_features, depth=50)

    print("\nFeature Importance:")
    for feature, value in result["importance"].items():
        print(f"{feature}: {value:.4f}")

    print("\nScientific Insight:")
    print(result["insight"])