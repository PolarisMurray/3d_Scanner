"""Edit scan processing settings here. Firmware angles are in src/main.cpp."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/sensor.npz"
GEOMETRY = ROOT / "models/geometry.json"
CALIBRATION_CSV = ROOT / "data/distance_voltage_measurements.csv"
OUTPUT = ROOT / "output"

SERIAL_PORT = "/dev/cu.usbmodem21401"
BAUD_RATE = 115200
RESET_WAIT_SECONDS = 2
SERIAL_IDLE_SECONDS = 15

CALIBRATION_MIN_DISTANCE_CM = 13.0
MAX_MESH_EDGE_CM = 4.0
# Match the scan steps in src/main.cpp; missing rows/columns stay empty.
PAN_STEP_DEG = 1.5
TILT_STEP_DEG = 1.5
# Odd window on the pan-tilt grid: 3 = median of valid 3x3 neighbors, 1 = off.
MEDIAN_WINDOW = 3

# +X right, +Y up, +Z forward; vectors are in cm at the 90/90 pose.
# Measure these on the actual bracket. Zero means unmeasured/ideal geometry.
PAN_CENTER = 90.0
TILT_CENTER = 90.0
PAN_TO_TILT_CM = [0.0, 0.0, 0.0]
TILT_TO_SENSOR_CM = [1.5, -4.0, 0.0]


def default_geometry():
    return {
        "pan_center_deg": PAN_CENTER,
        "tilt_center_deg": TILT_CENTER,
        "pan_to_tilt_cm": PAN_TO_TILT_CM,
        "tilt_to_sensor_cm": TILT_TO_SENSOR_CM,
        "parameters": [0.0] * 7,  # order: geometry.PARAMETER_NAMES
        "calibrated": False,
    }
