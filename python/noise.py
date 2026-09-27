"""Static noise test: raw readings at 90/90 with servos holding, then released.

Point the scanner at a flat wall or board 25-50 cm away and keep everything still.
Compares the old per-point method (mean of 10 readings in 50 ms) with the new
firmware method (median of 7 readings 40 ms apart), in volts and in cm.
"""

import numpy as np

from acquisition import acquire
from sensor import SensorModel

PHASES = {1: "servos holding", 0: "servos released"}
OLD_WINDOW_MS = 50
NEW_SAMPLES, NEW_SPACING_MS = 7, 40


def robust_sd(values):
    values = np.asarray(values)
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def old_method(ms, volts):
    blocks = (ms - ms[0]) // OLD_WINDOW_MS
    return np.array([volts[blocks == b].mean() for b in np.unique(blocks)[:-1]])


def new_method(ms, volts):
    points, start = [], ms[0]
    span = (NEW_SAMPLES - 1) * NEW_SPACING_MS
    while start + span <= ms[-1]:
        picks = np.searchsorted(ms, start + NEW_SPACING_MS * np.arange(NEW_SAMPLES))
        points.append(np.median(volts[picks]))
        start += span + NEW_SPACING_MS
    return np.array(points)


def cm_per_volt(sensor, volts):
    """Local slope of the distance model, to express voltage noise as distance."""
    return abs(float(np.diff(sensor.predict([volts - 0.02, volts + 0.02]))[0])) / 0.04


def run(port, output):
    frame = acquire(port, output, command=b"n", columns=["phase", "ms", "voltage_V"])
    sensor = SensorModel()
    print(f"Saved {output}")
    for phase, name in PHASES.items():
        part = frame[frame.phase == phase]
        if len(part) < 100:
            print(f"{name}: too few samples")
            continue
        ms, volts = part.ms.to_numpy(float), part.voltage_V.to_numpy(float)
        level = float(np.median(volts))
        inside = bool(sensor.measured_voltage(np.array([level]))[0])
        scale = cm_per_volt(sensor, level) if inside else np.nan
        where = (f"{float(sensor.predict([level])[0]):.1f} cm, {scale:.1f} cm/V" if inside
                 else "outside the model's voltage range: move closer, 20-60 cm")
        print(f"\n{name}: median {level:.3f} V ({where})")
        for label, values in (("single reading", volts),
                              ("old: mean 10 in 50 ms", old_method(ms, volts)),
                              ("new: median 7 x 40 ms", new_method(ms, volts))):
            sd = robust_sd(values)
            print(f"  {label:<24} sd {sd:.4f} V = {sd * scale:5.2f} cm  "
                  f"(max jump {np.ptp(values):.3f} V, n={len(values)})")
