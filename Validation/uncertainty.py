import numpy as np


def conformal_quantiles(y_true, y_pred, alpha=0.1):
    residuals = np.abs(y_true - y_pred)
    return np.quantile(residuals, 1 - alpha, axis=0)


def conformal_bounds(prediction, quantiles):
    lower = prediction - quantiles
    upper = prediction + quantiles

    return lower, upper