"""Fit flat-board scans using Taichi autodiff + bounded L-BFGS-B, and check the result."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from common import rms, robust_sigma, save_figure
from geometry import (N_PARAMETERS, PARAMETER_NAMES, RANGE_BIAS, RANGE_SCALE, TILT_LAG,
                      Geometry, default_geometry, fit_plane, load_geometry, save_geometry)
from sensor import SensorModel
from settings import GEOMETRY, OUTPUT

PLANE_COLUMNS = ["nx", "ny", "nz", "plane_offset_cm"]
ANGLE_BOUNDS = [(-10, 10), (-10, 10), (-20, 20), (-20, 20)]
RANGE_BOUNDS = [(-10, 10), (-5, 5)]
BOARD_SHIFT_CM = 10
# Smallest useful precision (deg, deg, %, %, %, cm). Worse-determined corrections stay 0.
USEFUL_SIGMA = [0.3, 0.3, 2.0, 2.0, 2.0, 0.5]


def labels(paths):
    """Names of every optimizer variable: model parameters, then one shift per board."""
    return PARAMETER_NAMES + [f"{path.name} distance" for path in paths]


def load_planes(paths, sensor):
    frame = pd.concat([pd.read_csv(path).assign(board=i) for i, path in enumerate(paths)],
                      ignore_index=True)
    # Use only voltages covered by the sensor's training data.
    frame = frame[sensor.measured_voltage(frame.voltage_V)].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No plane samples inside the measured sensor voltage domain")
    if "tilt_dir" not in frame:
        frame["tilt_dir"] = 0.0
    samples = np.column_stack([frame.pan_deg, frame.tilt_deg,
                               sensor.predict(frame.voltage_V), frame.tilt_dir])
    return frame, samples


def residuals(engine, x, planes, boards):
    """Signed distance (cm) of every reconstructed point from its board plane."""
    xyz = engine.points(x)
    return np.sum(xyz * planes[:, :3], axis=1) - planes[:, 3] - x[N_PARAMETERS:][boards]


def spread(engine, x, free, planes, boards, step=1e-4):
    """Gauss-Newton 1-sigma errors and correlations; assumes independent sample noise."""
    if not free:
        return np.empty(0), np.empty((0, 0))
    columns = []
    for i in free:
        dx = np.zeros_like(x)
        dx[i] = step
        columns.append((residuals(engine, x + dx, planes, boards)
                        - residuals(engine, x - dx, planes, boards)) / (2 * step))
    jacobian = np.column_stack(columns)
    sigma = robust_sigma(residuals(engine, x, planes, boards))
    inverse = np.linalg.pinv(jacobian, rcond=1e-8)
    covariance = sigma ** 2 * (inverse @ inverse.T)
    std = np.sqrt(np.maximum(np.diag(covariance), 0))
    # A null direction means unknown, not zero uncertainty from the pseudoinverse.
    _, singular, vectors = np.linalg.svd(jacobian, full_matrices=True)
    rank = np.count_nonzero(singular > singular[0] * 1e-8)
    unknown = np.sum(vectors[rank:] ** 2, axis=0) > 1e-8
    std[unknown] = np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        return std, covariance / np.outer(std, std)


def fit_geometry(samples, planes, boards, config, fit_range=False):
    """fit_range=False: each board distance is fitted, range correction stays fixed.
    fit_range=True: board distances are trusted as measured, range scale/bias are fitted.
    The tilt lag always stays at its configured value. Corrections the boards cannot
    determine to USEFUL_SIGMA are locked at their start value, worst first."""
    boards = np.asarray(boards)
    count = int(boards.max()) + 1
    engine = Geometry(samples, config, boards)
    engine.planes.from_numpy(np.asarray(planes, dtype=np.float64))
    start = np.concatenate([config["parameters"], np.zeros(count)])

    def fixed(i):
        return (start[i], start[i])

    # Bounds per variable; equal low/high bounds hold that variable fixed.
    bounds = list(ANGLE_BOUNDS)
    bounds += RANGE_BOUNDS if fit_range else [fixed(RANGE_SCALE), fixed(RANGE_BIAS)]
    bounds.append(fixed(TILT_LAG))  # measured separately by `scanner.py backlash`
    bounds += [(0.0, 0.0) if fit_range else (-BOARD_SHIFT_CM, BOARD_SHIFT_CM)] * count

    locked = {}
    while True:
        result = minimize(engine.objective, start, jac=True, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
        free = [i for i, (low, high) in enumerate(bounds) if low < high]
        std, correlation = spread(engine, result.x, free, planes, boards)
        # Uncertainty relative to what would be useful; > 1 means the boards cannot tell.
        ratio = {i: std[free.index(i)] / USEFUL_SIGMA[i] for i in free if i < len(USEFUL_SIGMA)}
        worst = max(ratio, key=ratio.get, default=None)
        if worst is None or ratio[worst] <= 1:
            break
        locked[worst] = float(std[free.index(worst)])
        bounds[worst] = fixed(worst)

    before = residuals(engine, start, planes, boards)
    if not fit_range:
        # Fair comparison: give the uncorrected geometry its best board distances too.
        before -= (np.bincount(boards, before) / np.maximum(np.bincount(boards), 1))[boards]
    at_limit = [i for i in free if np.isclose(result.x[i], bounds[i]).any()]
    return SimpleNamespace(result=result, before=before,
                           after=residuals(engine, result.x, planes, boards), free=free,
                           std=std, correlation=correlation, locked=locked, at_limit=at_limit)


def report(fit, frame, paths, fit_range):
    names = labels(paths)
    print("Parameters (+- approx. 1 sigma from the fit data):")
    for i, name in enumerate(PARAMETER_NAMES):
        if i in fit.free:
            note = f" +- {fit.std[fit.free.index(i)]:.3f}"
        elif i in fit.locked:
            note = f"  (kept: boards give only +- {fit.locked[i]:.2f})"
        else:
            note = "  (fixed)"
        print(f"  {name:20s} {fit.result.x[i]:+8.3f}{note}")
    if not fit_range:
        for k, path in enumerate(paths):
            rows = frame[frame.board == k]
            if rows.empty:
                print(f"  {path.name}: no samples in the sensor voltage domain")
                continue
            nz, offset = rows.nz.iloc[0], rows.plane_offset_cm.iloc[0]
            print(f"  {path.name}: board Z entered {offset / nz:.2f} cm, "
                  f"fitted {(offset + fit.result.x[N_PARAMETERS + k]) / nz:.2f} cm")
    for a in range(len(fit.free)):
        for b in range(a + 1, len(fit.free)):
            if abs(fit.correlation[a, b]) > 0.95:
                print(f"  Warning: {names[fit.free[a]]} and {names[fit.free[b]]} can substitute "
                      f"for each other (r={fit.correlation[a, b]:+.2f}); add tilted boards.")


def plot_residuals(frame, before, after, paths, show):
    limit = max(np.percentile(np.abs(before), 98), 0.1)
    fig, axes = plt.subplots(len(paths), 2, figsize=(11, 3.4 * len(paths)),
                             squeeze=False, layout="constrained")
    for k, path in enumerate(paths):
        rows = (frame.board == k).to_numpy()
        for ax, residual, label in zip(axes[k], (before, after), ("Before", "After")):
            image = ax.scatter(frame.pan_deg[rows], frame.tilt_deg[rows], c=residual[rows],
                               cmap="RdBu_r", vmin=-limit, vmax=limit, marker="s", s=28)
            ax.invert_yaxis()
            ax.set(xlabel="Pan command (deg)", ylabel="Tilt command (deg)",
                   title=f"{path.name} {label}: RMSE {rms(residual[rows]):.3f} cm")
    fig.colorbar(image, ax=axes, label="Distance from plane (cm)")
    save_figure(fig, OUTPUT / "geometry_calibration.png", show)
    print(f"Saved {OUTPUT / 'geometry_calibration.png'}")


def calibrate(paths, fit_range=False, show=True):
    sensor = SensorModel()
    frame, samples = load_planes(paths, sensor)
    planes = frame[PLANE_COLUMNS].to_numpy()
    boards = frame.board.to_numpy()
    config = default_geometry()
    # Keep the lag from `scanner.py backlash`; flat boards constrain it only weakly.
    saved = load_geometry(GEOMETRY)
    config["parameters"][TILT_LAG] = saved["parameters"][TILT_LAG]
    if "tilt_lag_source" in saved:
        config["tilt_lag_source"] = saved["tilt_lag_source"]

    fit = fit_geometry(samples, planes, boards, config, fit_range)
    print(f"Plane RMSE: {rms(fit.before):.3f} -> {rms(fit.after):.3f} cm (fit data)")
    print(f"Optimizer: {fit.result.message}")
    report(fit, frame, paths, fit_range)
    plot_residuals(frame, fit.before, fit.after, paths, show)

    # Save only a converged, interior solution that actually corrects something.
    if not fit.result.success:
        raise RuntimeError("Calibration did not converge; geometry.json not changed")
    if fit.at_limit:
        names = labels(paths)
        raise RuntimeError(", ".join(names[i] for i in fit.at_limit) + " reached the optimizer "
                           "limit; geometry.json not changed. Check the board setup and noise.")
    if not any(i < N_PARAMETERS for i in fit.free):
        print("These boards determine no geometry correction; geometry.json not changed.")
        return
    config.update(parameters=fit.result.x[:N_PARAMETERS].tolist(), calibrated=True,
                  fit_mode="measured board Z + range" if fit_range else "fitted board Z",
                  kept_at_zero=[PARAMETER_NAMES[i] for i in fit.locked],
                  source_files=[str(path.resolve()) for path in paths],
                  parameter_names=PARAMETER_NAMES, fit_rmse_cm=rms(fit.after))
    save_geometry(config, GEOMETRY)
    print(f"Saved {GEOMETRY}")


def check(paths):
    """Score boards not used for fitting: flatness, tilt of the plane, and Z error."""
    sensor = SensorModel()
    configs = {"default": default_geometry()}
    if GEOMETRY.exists():
        configs["geometry.json"] = load_geometry(GEOMETRY)
    for path in paths:
        frame, samples = load_planes([path], sensor)
        expected = frame[PLANE_COLUMNS[:3]].iloc[0].to_numpy()
        z = frame.plane_offset_cm.iloc[0] / expected[2]
        print(f"{path.name}: {len(frame)} samples, board Z entered as {z:.2f} cm")
        for label, config in configs.items():
            xyz = Geometry(samples, config).points()
            center, normal, _ = fit_plane(xyz)
            distance = (xyz - center) @ normal
            angle = np.degrees(np.arccos(min(1.0, abs(normal @ expected))))
            print(f"  {label:13s} flatness RMSE {rms(distance):.3f} cm "
                  f"(robust {robust_sigma(distance):.3f}), "
                  f"plane tilt error {angle:.2f} deg, "
                  f"Z error {(normal @ center) / normal[2] - z:+.2f} cm")
