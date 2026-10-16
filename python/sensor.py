"""Voltage-to-distance MLP, evaluated in plain NumPy from a portable .npz file."""

import numpy as np

from settings import MODEL


class SensorModel:
    """Weights written by train_mlp.py: input/output scalers, layers w0.. / b0..,
    and the voltage range seen in training (outside it the model extrapolates)."""

    def __init__(self, path=MODEL):
        with np.load(path) as data:
            self.weights = {name: data[name] for name in data.files}
        self.n_layers = sum(name[0] == "w" and name[1:].isdigit() for name in self.weights)

    def predict(self, voltage):
        w = self.weights
        x = (np.asarray(voltage, dtype=float).reshape(-1, 1) - w["x_mean"]) / w["x_scale"]
        for i in range(self.n_layers):
            x = x @ w[f"w{i}"] + w[f"b{i}"]
            if i < self.n_layers - 1:
                x = np.maximum(x, 0)  # ReLU on hidden layers, linear output
        return (x * w["y_scale"] + w["y_mean"]).ravel()

    def measured_voltage(self, voltage):
        """True where the voltage lies inside the calibration data."""
        low, high = self.weights["voltage_range"]
        return np.asarray((voltage >= low) & (voltage <= high))
