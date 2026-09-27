"""Taichi pan/tilt kinematics and automatic derivatives of plane error."""

import atexit
import json

import numpy as np
import taichi as ti

from settings import GEOMETRY, default_geometry

PARAMETER_NAMES = ["pan_zero_deg", "tilt_zero_deg", "pan_scale_percent",
                   "tilt_scale_percent", "range_scale_percent", "range_bias_cm",
                   "tilt_lag_deg"]
N_PARAMETERS = len(PARAMETER_NAMES)
_initialized = False


def initialize():
    global _initialized
    if not _initialized:
        ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, log_level=ti.ERROR)
        # Release native fields before Python tears down modules on macOS.
        atexit.register(ti.reset)
        _initialized = True


def load_geometry(path=GEOMETRY):
    return json.loads(path.read_text()) if path.exists() else default_geometry()


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
        limit = max(clip * 1.4826 * np.median(np.abs(residual[keep])), 0.05)
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
        a = self.samples[i]
        # Servo stops short of the command in its direction of travel.
        tilt = a[1] - a[3] * self.params[6]
        theta = ((a[0] - self.pan_center) * (1 + self.params[2] / 100)
                 + self.params[0]) * (np.pi / 180)
        phi = ((tilt - self.tilt_center) * (1 + self.params[3] / 100)
               + self.params[1]) * (np.pi / 180)
        distance = a[2] * (1 + self.params[4] / 100) + self.params[5]
        sensor = self.arm[1] + ti.Vector([0.0, 0.0, distance])
        # Tilt about +X, then pan about +Y; smaller tilt points upward.
        tilted = ti.Vector([sensor[0],
                            ti.cos(phi) * sensor[1] - ti.sin(phi) * sensor[2],
                            ti.sin(phi) * sensor[1] + ti.cos(phi) * sensor[2]])
        p = self.arm[0] + tilted
        return ti.Vector([ti.cos(theta) * p[0] + ti.sin(theta) * p[2], p[1],
                          -ti.sin(theta) * p[0] + ti.cos(theta) * p[2]])

    @ti.kernel
    def reconstruct(self):
        for i in range(self.n):
            self.xyz[i] = self.point(i)

    @ti.kernel
    def plane_loss(self):
        for i in range(self.n):
            p = self.point(i)
            plane = self.planes[i]
            residual = (p.dot(ti.Vector([plane[0], plane[1], plane[2]])) - plane[3]
                        - self.shift[self.board[i]])
            # Smooth robust loss: large outliers have less influence (delta = 1 cm).
            self.loss[None] += 2 * (ti.sqrt(1 + residual * residual) - 1) / self.n
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
