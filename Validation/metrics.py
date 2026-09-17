import numpy as np


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred), axis=0)


def correlation(y_true, y_pred):
    correlations = []

    for i in range(y_true.shape[1]):
        correlations.append(
            np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
        )

    return np.array(correlations)


def bias(y_true, y_pred):
    return np.mean(y_pred - y_true, axis=0)