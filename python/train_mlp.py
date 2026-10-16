"""Train the voltage-to-distance MLP from the measured calibration CSV,
or (--errors-only) evaluate the saved model without retraining."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from common import rms, save_figure
from sensor import SensorModel
from settings import CALIBRATION_CSV, CALIBRATION_MIN_DISTANCE_CM, MODEL, OUTPUT

ERROR_BANDS_CM = [13, 30, 45, 60, 83]


def load_calibration(csv_path):
    """Calibration rows; very close distances are dropped because the sensor folds back."""
    frame = pd.read_csv(csv_path)
    return frame[frame.distance_cm >= CALIBRATION_MIN_DISTANCE_CM].reset_index(drop=True)


def fit(x, y):
    sx, sy = StandardScaler(), StandardScaler()
    model = MLPRegressor(hidden_layer_sizes=(32, 32), solver="lbfgs",
                         max_iter=10000, max_fun=100000, random_state=42)
    model.fit(sx.fit_transform(x), sy.fit_transform(y).ravel())
    return model, sx, sy


def save_weights(model, sx, sy, voltage, path):
    """Store everything SensorModel needs as plain arrays (no scikit-learn at runtime)."""
    weights = {f"w{i}": w for i, w in enumerate(model.coefs_)}
    weights.update({f"b{i}": b for i, b in enumerate(model.intercepts_)})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **weights, x_mean=sx.mean_, x_scale=sx.scale_,
             y_mean=sy.mean_, y_scale=sy.scale_, voltage_range=[voltage.min(), voltage.max()])


def train(csv_path=CALIBRATION_CSV, output=MODEL, show=True):
    frame = load_calibration(csv_path)
    x = frame[["voltage_mean"]].to_numpy()
    y = frame.distance_cm.to_numpy()

    # Hold out whole distances (20 %) to measure interpolation error honestly.
    train_d, _ = train_test_split(frame.distance_cm.unique(), test_size=0.2, random_state=42)
    use = frame.distance_cm.isin(train_d).to_numpy()
    model, sx, sy = fit(x[use], y[use].reshape(-1, 1))
    predicted = sy.inverse_transform(model.predict(sx.transform(x[~use])).reshape(-1, 1)).ravel()
    error = predicted - y[~use]
    print(f"Held-out MAE: {np.abs(error).mean():.3f} cm")
    print(f"Held-out RMSE: {rms(error):.3f} cm")

    # The saved model uses every distance.
    model, sx, sy = fit(x, y.reshape(-1, 1))
    save_weights(model, sx, sy, x, output)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    grid = np.linspace(x.min(), x.max(), 300)
    axes[0].scatter(x.ravel(), y, s=15, label="Measured")
    axes[0].plot(grid, SensorModel(output).predict(grid), label="MLP")
    axes[0].set(xlabel="Voltage (V)", ylabel="Distance (cm)", title="Sensor calibration")
    axes[0].legend()
    axes[1].scatter(y[~use], predicted)
    axes[1].plot([y.min(), y.max()], [y.min(), y.max()], "k--")
    axes[1].set(xlabel="Measured (cm)", ylabel="Predicted (cm)", title="Held-out distances")
    fig.tight_layout()
    save_figure(fig, OUTPUT / "sensor_training.png", show)
    print(f"Saved {output}")


def plot_errors(csv_path=CALIBRATION_CSV, model_path=MODEL, show=True):
    """Plot and tabulate predicted - true distance on every calibration point. Does not retrain."""
    frame = load_calibration(csv_path)
    true = frame.distance_cm.to_numpy()
    predicted = SensorModel(model_path).predict(frame.voltage_mean)
    error = predicted - true

    # Table 1: error statistics per distance band
    band_labels = [f"{low}-{high - 1} cm" for low, high in zip(ERROR_BANDS_CM, ERROR_BANDS_CM[1:])]
    bands = pd.cut(true, ERROR_BANDS_CM, right=False, labels=band_labels)
    table = (pd.DataFrame({"band": bands, "abs": np.abs(error), "err": error})
             .groupby("band", observed=True)
             .agg(points=("abs", "size"), MAE_cm=("abs", "mean"),
                  max_cm=("abs", "max"), bias_cm=("err", "mean"))
             .round(3))
    table.loc["All"] = [len(error), np.abs(error).mean().round(3),
                        np.abs(error).max().round(3), error.mean().round(3)]
    print("\nError by distance band")
    print(table.to_string())

    # Table 2: the five worst points
    worst = pd.DataFrame({"true_cm": true, "voltage_V": frame.voltage_mean.to_numpy(),
                          "predicted_cm": predicted.round(2), "error_cm": error.round(3)})
    worst = worst.reindex(worst.error_cm.abs().sort_values(ascending=False).index).head(5)
    print("\nFive largest errors")
    print(worst.to_string(index=False))
    print(f"\nRMSE {rms(error):.3f} cm")

    # Figure: error plot on the left, band table on the right
    fig, (ax, tab) = plt.subplots(1, 2, figsize=(12, 4.5),
                                  gridspec_kw={"width_ratios": [1.6, 1]})
    ax.axhline(0, color="gray", linewidth=1)
    ax.plot(true, error, "-o", markersize=3)
    ax.set(xlabel="True distance (cm)", ylabel="Predicted - true (cm)",
           title="MLP error on calibration points")
    tab.axis("off")
    cells = tab.table(cellText=table.reset_index().astype(str).values,
                      colLabels=["Band", "Points", "MAE (cm)", "Max (cm)", "Bias (cm)"],
                      loc="center", cellLoc="center")
    cells.scale(1, 1.6)
    tab.set_title("Error by distance band")
    fig.tight_layout()
    save_figure(fig, OUTPUT / "sensor_error.png", show)
    table.to_csv(OUTPUT / "sensor_error_table.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CALIBRATION_CSV)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--errors-only", action="store_true",
                        help="Plot the saved model's error without retraining")
    args = parser.parse_args()
    if args.errors_only:
        plot_errors(args.csv, show=not args.no_show)
    else:
        train(args.csv, show=not args.no_show)
