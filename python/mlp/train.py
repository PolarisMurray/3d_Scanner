"""Train the MLP: measure held-out error, then fit on all data and save."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from common import rms, save_figure
from sensor import SensorModel
from settings import CALIBRATION_CSV, MODEL, OUTPUT

from .data import load_calibration
from .network import SEED, fit, save

HOLDOUT_FRACTION = 0.2


def train(csv_path = CALIBRATION_CSV, output = MODEL, show = True):
    voltage, distance = load_calibration(csv_path)

    # 1. Hold out whole distances to measure interpolation error honestly.
    _, test_distances = train_test_split(
        np.unique(distance),
        test_size = HOLDOUT_FRACTION,
        random_state = SEED,
    )
    test = np.isin(distance, test_distances)

    held_out = SensorModel(weights = fit(voltage[~test], distance[~test]))
    predicted = held_out.predict(voltage[test])
    error = predicted - distance[test]

    print(f"Held-out MAE:  {np.abs(error).mean():.3f} cm")
    print(f"Held-out RMSE: {rms(error):.3f} cm")

    # 2. The saved model is trained on every distance.
    weights = fit(voltage, distance)
    save(weights, output)
    print(f"Saved {output}")

    # 3. Plot the fitted curve and the held-out predictions.
    model = SensorModel(weights = weights)
    plot_training(voltage, distance, model, distance[test], predicted, show)


def plot_training(voltage, distance, model, test_true, test_predicted, show):
    fig, (left, right) = plt.subplots(1, 2, figsize = (11, 4))

    grid = np.linspace(voltage.min(), voltage.max(), 300)
    left.scatter(voltage, distance, s = 15, label = "Measured")
    left.plot(grid, model.predict(grid), label = "MLP")
    left.set(xlabel = "Voltage (V)", ylabel = "Distance (cm)", title = "Sensor calibration")
    left.legend()

    line = [distance.min(), distance.max()]
    right.scatter(test_true, test_predicted)
    right.plot(line, line, "k--")
    right.set(xlabel = "Measured (cm)", ylabel = "Predicted (cm)", title = "Held-out distances")

    fig.tight_layout()
    save_figure(fig, OUTPUT / "sensor_training.png", show)
