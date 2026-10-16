"""Taichi pan/tilt kinematics and automatic derivatives of plane error."""

import atexit
import json

import numpy as np
import taichi as ti

from common import robust_sigma, write_json
from settings import GEOMETRY, PAN_CENTER, PAN_TO_TILT_CM, TILT_CENTER, TILT_TO_SENSOR_CM

# Correction parameters, in this order in geometry.json "parameters".
PARAMETER_NAMES = ["pan_zero_deg", "tilt_zero_deg", "pan_scale_percent",
                   "tilt_scale_percent", "range_scale_percent", "range_bias_cm",
                   "tilt_lag_deg"]
N_PARAMETERS = len(PARAMETER_NAMES)
PAN_ZERO, TILT_ZERO, PAN_SCALE, TILT_SCALE, RANGE_SCALE, RANGE_BIAS, TILT_LAG = range(N_PARAMETERS)
_initialized = False


def initialize():
    global _initialized
    if not _initialized:
        ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, log_level=ti.ERROR)
        # Release native fields before Python tears down modules on macOS.
        atexit.register(ti.reset)
        _initialized = True


def default_geometry():
    return {
        "pan_center_deg": PAN_CENTER,
        "tilt_center_deg": TILT_CENTER,
        "pan_to_tilt_cm": PAN_TO_TILT_CM,
        "tilt_to_sensor_cm": TILT_TO_SENSOR_CM,
        "parameters": [0.0] * N_PARAMETERS,
        "calibrated": False,
    }


def load_geometry(path=GEOMETRY):
    return json.loads(path.read_text()) if path.exists() else default_geometry()


def save_geometry(config, path=GEOMETRY):
    write_json(path, config)


def corrected_distance(distance, parameters):
    """Range scale/bias correction; Geometry.point applies the same formula."""
    return distance * (1 + parameters[RANGE_SCALE] / 100) + parameters[RANGE_BIAS]


def fit_plane(xyz, clip=3.0, rounds=10):
    """Dominant plane z = a + b x + c y facing the scanner, refit after MAD outlier clipping.

    Range noise lies mostly along Z; an orthogonal fit can flip on small noisy patches.
    Returns the Z-axis intercept point, the unit normal toward the scanner, and inliers.
    """
    design = np.column_stack([np.ones(len(xyz)), xyz[:, 0], xyz[:, 1]])
    keep = np.ones(len(xyz), dtype=bool)
    for _ in range(rounds):
        a, b, c = np.linalg.lstsq(design[keep], xyz[keep, 2], rcond=None)[0]
        residual = xyz[:, 2] - design @ [a, b, c]
        limit = max(clip * robust_sigma(residual[keep]), 0.05)
        inliers = np.abs(residual) <= limit
        if inliers.sum() < 3 or np.array_equal(inliers, keep):
            break
        keep = inliers
    normal = np.array([b, c, -1.0])
    return np.array([0.0, 0.0, a]), normal / np.linalg.norm(normal), keep


@ti.data_oriented
class Geometry:
    """samples: pan, tilt, measured distance, tilt direction (+1/-1, 0 = unknown)."""

    def __init__(self, samples, config, boards=None):
        initialize()
        self.n = len(samples)
        boards = np.zeros(self.n, dtype=np.int32) if boards is None else np.asarray(boards)
        self.pan_center = config["pan_center_deg"]
        self.tilt_center = config["tilt_center_deg"]
        self.arm = ti.Vector.field(3, ti.f64, shape=2)
        self.arm.from_numpy(np.array([config["pan_to_tilt_cm"],
                                       config["tilt_to_sensor_cm"]], dtype=np.float64))
        self.samples = ti.Vector.field(4, ti.f64, shape=self.n)
        self.samples.from_numpy(np.asarray(samples, dtype=np.float64))
        self.board = ti.field(ti.i32, shape=self.n)
        self.board.from_numpy(boards.astype(np.int32))
        self.params = ti.field(ti.f64, shape=N_PARAMETERS, needs_grad=True)
        self.params.from_numpy(np.array(config["parameters"], dtype=np.float64))
        # Unknown distance of each flat board along its normal (calibration only).
        self.shift = ti.field(ti.f64, shape=int(boards.max()) + 1, needs_grad=True)
        self.xyz = ti.Vector.field(3, ti.f64, shape=self.n)
        self.planes = ti.Vector.field(4, ti.f64, shape=self.n)
        self.loss = ti.field(ti.f64, shape=(), needs_grad=True)

    @ti.func
    def point(self, i):
        pan, tilt, distance, direction = self.samples[i]
        p = ti.static(self.params)
        # Servo stops short of the command in its direction of travel.
        tilt -= direction * p[TILT_LAG]
        theta = ((pan - self.pan_center) * (1 + p[PAN_SCALE] / 100) + p[PAN_ZERO]) * (np.pi / 180)
        phi = ((tilt - self.tilt_center) * (1 + p[TILT_SCALE] / 100) + p[TILT_ZERO]) * (np.pi / 180)
        distance = distance * (1 + p[RANGE_SCALE] / 100) + p[RANGE_BIAS]
        sensor = self.arm[1] + ti.Vector([0.0, 0.0, distance])
        # Tilt about +X, then pan about +Y; smaller tilt points upward.
        tilted = ti.Vector([sensor[0],
                            ti.cos(phi) * sensor[1] - ti.sin(phi) * sensor[2],
                            ti.sin(phi) * sensor[1] + ti.cos(phi) * sensor[2]])
        q = self.arm[0] + tilted
        return ti.Vector([ti.cos(theta) * q[0] + ti.sin(theta) * q[2], q[1],
                          -ti.sin(theta) * q[0] + ti.cos(theta) * q[2]])

    @ti.kernel
    def reconstruct(self):
        for i in range(self.n):
            self.xyz[i] = self.point(i)

    @ti.kernel
    def plane_loss(self):
        for i in range(self.n):
            plane = self.planes[i]
            normal = ti.Vector([plane[0], plane[1], plane[2]])
            residual = self.point(i).dot(normal) - plane[3] - self.shift[self.board[i]]
            # Smooth robust loss: large outliers have less influence (delta = 1 cm).
            self.loss[None] += 2 * (ti.sqrt(1 + residual * residual) - 1) / self.n
        # Tiny ridge term keeps parameters the boards cannot see near zero.
        for j in range(N_PARAMETERS):
            self.loss[None] += 0.00001 * self.params[j] ** 2

    def objective(self, x):
        """x = model parameters followed by one distance shift per board."""
        self.params.from_numpy(np.asarray(x[:N_PARAMETERS], dtype=np.float64))
        self.shift.from_numpy(np.asarray(x[N_PARAMETERS:], dtype=np.float64))
        with ti.ad.Tape(loss=self.loss):
            self.plane_loss()
        return float(self.loss[None]), np.concatenate([self.params.grad.to_numpy(),
                                                       self.shift.grad.to_numpy()])

    def points(self, parameters=None):
        if parameters is not None:
            self.params.from_numpy(np.asarray(parameters[:N_PARAMETERS], dtype=np.float64))
        self.reconstruct()
        return self.xyz.to_numpy()
