"""Train the sensor curve using the existing measured calibration CSV."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from sensor import SensorModel
from settings import CALIBRATION_CSV, CALIBRATION_MIN_DISTANCE_CM, MODEL, OUTPUT


def fit(x, y):
    sx, sy = StandardScaler(), StandardScaler()
    model = MLPRegressor(hidden_layer_sizes=(32, 32), solver="lbfgs",
                         max_iter=10000, max_fun=100000, random_state=42)
    model.fit(sx.fit_transform(x), sy.fit_transform(y).ravel())
    return model, sx, sy


def train(csv_path=CALIBRATION_CSV, output=MODEL, show=True):
    frame = pd.read_csv(csv_path)
    frame = frame[frame.distance_cm >= CALIBRATION_MIN_DISTANCE_CM]
    x = frame[["voltage_mean"]].to_numpy()
    y = frame[["distance_cm"]].to_numpy()
    train_d, _ = train_test_split(frame.distance_cm.unique(), test_size=0.2, random_state=42)
    use = frame.distance_cm.isin(train_d).to_numpy()
    model, sx, sy = fit(x[use], y[use])
    predicted = sy.inverse_transform(model.predict(sx.transform(x[~use])).reshape(-1, 1))
    print(f"Held-out MAE: {mean_absolute_error(y[~use], predicted):.3f} cm")
    print(f"Held-out RMSE: {np.sqrt(mean_squared_error(y[~use], predicted)):.3f} cm")

    model, sx, sy = fit(x, y)
    weights = {f"w{i}": w for i, w in enumerate(model.coefs_)}
    weights.update({f"b{i}": b for i, b in enumerate(model.intercepts_)})
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **weights, x_mean=sx.mean_, x_scale=sx.scale_,
             y_mean=sy.mean_, y_scale=sy.scale_, voltage_range=[x.min(), x.max()])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    grid = np.linspace(x.min(), x.max(), 300)
    axes[0].scatter(x.ravel(), y.ravel(), s=15, label="Measured")
    axes[0].plot(grid, SensorModel(output).predict(grid), label="MLP")
    axes[0].set(xlabel="Voltage (V)", ylabel="Distance (cm)", title="Sensor calibration")
    axes[0].legend()
    axes[1].scatter(y[~use].ravel(), predicted.ravel())
    axes[1].plot([y.min(), y.max()], [y.min(), y.max()], "k--")
    axes[1].set(xlabel="Measured (cm)", ylabel="Predicted (cm)", title="Held-out distances")
    OUTPUT.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "sensor_training.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CALIBRATION_CSV)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    train(args.csv, show=not args.no_show)
