"""Small voltage-to-distance MLP. Store weights in a portable NumPy file."""

import numpy as np

from settings import MODEL


class SensorModel:
    def __init__(self, path=MODEL):
        with np.load(path) as data:
            self.weights = {name: data[name] for name in data.files}

    def predict(self, voltage):
        w = self.weights
        x = (np.asarray(voltage).reshape(-1, 1) - w["x_mean"]) / w["x_scale"]
        for i in range(3):
            x = x @ w[f"w{i}"] + w[f"b{i}"]
            if i < 2:
                x = np.maximum(x, 0)
        return (x * w["y_scale"] + w["y_mean"]).ravel()

    def measured_voltage(self, voltage):
        low, high = self.weights["voltage_range"]
        return np.asarray((voltage >= low) & (voltage <= high))
