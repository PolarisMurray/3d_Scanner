"""Fit the MLP and save its weights as plain arrays (no scikit-learn needed at scan time)."""

import numpy as np
from sklearn.neural_network import MLPRegressor

HIDDEN_LAYERS = (32, 32)
SEED = 42


def fit(voltage, distance):
    """Fit on standardized voltage and distance.

    Returns the weights in the format sensor.SensorModel reads.
    """
    x_mean, x_scale = voltage.mean(), voltage.std()
    y_mean, y_scale = distance.mean(), distance.std()

    x = ((voltage - x_mean) / x_scale).reshape(-1, 1)
    y = (distance - y_mean) / y_scale

    mlp = MLPRegressor(
        hidden_layer_sizes = HIDDEN_LAYERS,
        solver = "lbfgs",
        max_iter = 10000,
        max_fun = 100000,
        random_state = SEED,
    )
    mlp.fit(x, y)

    weights = {}
    for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        weights[f"w{i}"] = w
        weights[f"b{i}"] = b

    weights["x_mean"] = x_mean
    weights["x_scale"] = x_scale
    weights["y_mean"] = y_mean
    weights["y_scale"] = y_scale
    weights["voltage_range"] = [voltage.min(), voltage.max()]
    return weights


def save(weights, path):
    path.parent.mkdir(parents = True, exist_ok = True)
    np.savez(path, **weights)
