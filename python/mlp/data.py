"""Load the measured calibration points."""

import pandas as pd

from settings import CALIBRATION_MIN_DISTANCE_CM


def load_calibration(csv_path):
    """Return voltage and distance arrays.

    Very close distances are dropped: there the sensor folds back.
    """
    frame = pd.read_csv(csv_path)
    frame = frame[frame.distance_cm >= CALIBRATION_MIN_DISTANCE_CM]

    voltage = frame.voltage_mean.to_numpy()
    distance = frame.distance_cm.to_numpy()
    return voltage, distance
