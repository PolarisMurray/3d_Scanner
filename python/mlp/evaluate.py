"""Error of the saved model on every calibration point. Does not retrain."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import rms, save_figure
from sensor import SensorModel
from settings import CALIBRATION_CSV, MODEL, OUTPUT

from .data import load_calibration

ERROR_BANDS_CM = [13, 30, 45, 60, 83]


def evaluate(csv_path = CALIBRATION_CSV, model_path = MODEL, show = True):
    voltage, distance = load_calibration(csv_path)
    error = SensorModel(model_path).predict(voltage) - distance

    table = band_table(distance, error)
    print(table.to_string())
    print(
        f"\nAll {len(error)} points: "
        f"MAE {np.abs(error).mean():.3f} cm, "
        f"max {np.abs(error).max():.3f} cm, "
        f"RMSE {rms(error):.3f} cm"
    )
    table.to_csv(OUTPUT / "sensor_error_table.csv")

    plot_error(distance, error, show)


def band_table(distance, error):
    """Points, mean / max absolute error and bias per distance band."""
    errors = pd.DataFrame({
        "band": pd.cut(distance, ERROR_BANDS_CM, right = False),
        "error": error,
        "abs": np.abs(error),
    })

    table = errors.groupby("band", observed = True).agg(
        points = ("abs", "size"),
        mae_cm = ("abs", "mean"),
        max_cm = ("abs", "max"),
        bias_cm = ("error", "mean"),
    )
    return table.round(3)


def plot_error(distance, error, show):
    fig, ax = plt.subplots(figsize = (8, 4))

    ax.axhline(0, color = "gray", linewidth = 1)
    ax.plot(distance, error, "-o", markersize = 3)
    ax.set(
        xlabel = "True distance (cm)",
        ylabel = "Predicted - true (cm)",
        title = "MLP error on calibration points",
    )

    fig.tight_layout()
    save_figure(fig, OUTPUT / "sensor_error.png", show)
