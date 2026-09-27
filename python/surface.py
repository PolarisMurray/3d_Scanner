"""Depth grid, optional median filter, relief height, mesh, plots and PLY."""

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from acquisition import load_raw
from geometry import Geometry, PARAMETER_NAMES, fit_plane, load_geometry
from sensor import SensorModel
from settings import MAX_MESH_EDGE_CM, MEDIAN_WINDOW, PAN_STEP_DEG, TILT_STEP_DEG

GRIDS = ["distance_cm", "z_cm", "height_cm"]


def scan_axis(angles, step):
    """Restore missing command positions so filtering never joins across a gap."""
    axis = []
    for value in np.sort(np.unique(angles)):
        if axis:
            axis.extend(np.arange(axis[-1] + step, value - 1e-8, step))
        axis.append(value)
    return np.asarray(axis)


def median_filter(grid, size):
    """Median of valid neighbors. Holes stay holes; the result is always a measured value."""
    if size <= 1 or grid.size == 0:
        return grid
    windows = sliding_window_view(np.pad(grid, size // 2, constant_values=np.nan), (size, size))
    windows = windows.reshape(*grid.shape, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows inside holes
        low, high = (np.nanquantile(windows, 0.5, axis=-1, method=m) for m in ("lower", "higher"))
    # Even counts: pick the middle value nearer the center instead of averaging two surfaces.
    median = np.where(np.abs(low - grid) <= np.abs(high - grid), low, high)
    sparse = np.isfinite(windows).sum(axis=-1) <= windows.shape[-1] // 2
    return np.where(np.isnan(grid) | sparse, grid, median)


def noise(grid):
    """Per-sample noise from second differences along tilt; robust to occasional edges."""
    second = (grid[2:] - 2 * grid[1:-1] + grid[:-2]).ravel()
    second = second[np.isfinite(second)]
    return float(1.4826 * np.median(np.abs(second)) / np.sqrt(6)) if len(second) else None


def mesh_faces(indices, xyz, edge_limit=MAX_MESH_EDGE_CM):
    """indices[row, col] = point index or -1. Connect only measured neighbors with short edges."""
    faces = []
    for row in range(indices.shape[0] - 1):
        for col in range(indices.shape[1] - 1):
            a, b = indices[row, col:col + 2]
            c, d = indices[row + 1, col:col + 2]
            for triangle in ((a, c, b), (b, c, d)):
                if min(triangle) < 0:
                    continue
                points = xyz[list(triangle)]
                edges = points - np.roll(points, 1, axis=0)
                if np.linalg.norm(edges, axis=1).max() <= edge_limit:
                    faces.append(triangle)
    return np.asarray(faces, dtype=int).reshape(-1, 3)


def relief(xyz):
    """Height above the dominant (base) plane, positive toward the scanner."""
    if len(xyz) < 10:
        return np.full(len(xyz), np.nan), None
    center, normal, _ = fit_plane(xyz)
    return (xyz - center) @ normal, (center, normal)


def write_ply(path, xyz, faces):
    with path.open("w") as stream:
        stream.write("ply\nformat ascii 1.0\ncomment coordinates in centimeters\n")
        stream.write(f"element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n")
        stream.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for point in xyz:
            stream.write(" ".join(f"{value:.6f}" for value in point) + "\n")
        for face in faces:
            stream.write("3 " + " ".join(map(str, face)) + "\n")


def placeholder(output, names, text):
    """Replace stale images when an output folder is reused."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, text, ha="center", va="center")
    ax.axis("off")
    for name in names:
        fig.savefig(output / name, dpi=150)
    plt.close(fig)


def grid_image(fig, ax, pan, tilt, values, title, label, **style):
    image = ax.pcolormesh(pan, tilt, np.ma.masked_invalid(values), shading="nearest", **style)
    ax.invert_yaxis()
    ax.set(xlabel="Pan command (deg)", ylabel="Tilt command (deg)", title=title)
    fig.colorbar(image, ax=ax, label=label)


def plot_results(cloud, pan, tilt, grids, faces, base, output, show):
    x, y, z, height = (cloud[name].to_numpy() for name in ("x_cm", "y_cm", "z_cm", "height_cm"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    scatter = axes[0].scatter(x, y, c=z, s=16, cmap="viridis")
    axes[0].set(xlabel="X (cm)", ylabel="Y (cm)", title="Front view: measured points")
    axes[0].set_aspect("equal", adjustable="box")
    fig.colorbar(scatter, ax=axes[0], label="Forward depth Z (cm)")
    grid_image(fig, axes[1], pan, tilt, grids["z_cm"], "Depth grid: blanks = discarded / missing",
               "Forward depth Z (cm)")
    fig.savefig(output / "depth.png", dpi=180)

    style = dict(cmap="viridis")
    if base is None:
        placeholder(output, ["relief.png"], "Too few points for a relief base plane")
    else:
        limit = max(float(np.percentile(np.abs(height), 98)), 0.2)
        style = dict(cmap="RdBu_r", vmin=-limit, vmax=limit)
        center, normal = base
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        if len(faces):
            axes[0].tripcolor(x, y, height, triangles=faces, shading="gouraud", **style)
        image = axes[0].scatter(x, y, c=height, s=5 if len(faces) else 16, **style)
        axes[0].set(xlabel="X (cm)", ylabel="Y (cm)", title="Front view, lag-corrected geometry")
        axes[0].set_aspect("equal", adjustable="box")
        fig.colorbar(image, ax=axes[0], label="Height above base plane (cm)")
        grid_image(fig, axes[1], pan, tilt, grids["height_cm"], "Relief on the scan grid",
                   "Height above base plane (cm)", **style)
        fig.suptitle(f"Relief: red = toward scanner. Base plane Z-intercept "
                     f"{(normal @ center) / normal[2]:.1f} cm, tilted "
                     f"{np.degrees(np.arccos(abs(normal[2]))):.1f} deg from facing the scanner")
        fig.savefig(output / "relief.png", dpi=180)

    fig3d = plt.figure(figsize=(9, 7))
    ax = fig3d.add_subplot(111, projection="3d")
    color = height if base is not None else z
    if len(faces):
        surface = ax.plot_trisurf(x, y, z, triangles=faces, cmap=style["cmap"],
                                  linewidth=0.1, alpha=0.85)
        surface.set_array(color[faces].mean(axis=1))
        surface.set_clim(style.get("vmin", z.min()), style.get("vmax", z.max()))
    ax.scatter(x, y, z, c=color, s=5, **style)
    xyz = np.column_stack([x, y, z])
    middle = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
    radius = max(float(np.ptp(xyz, axis=0).max()) / 2, 0.5)
    ax.set_xlim(middle[0] - radius, middle[0] + radius)
    ax.set_ylim(middle[1] - radius, middle[1] + radius)
    ax.set_zlim(middle[2] - radius, middle[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=15, azim=-65, vertical_axis="y")
    ax.set(xlabel="X (cm)", ylabel="Y (cm)", zlabel="Z (cm)",
           title="Taichi reconstruction, colored by "
                 + ("relief height" if base is not None else "depth Z"))
    fig3d.tight_layout()
    fig3d.savefig(output / "surface.png", dpi=180)
    if show:
        plt.show()
    plt.close("all")


def reconstruct(raw_path, output, show=True, median=MEDIAN_WINDOW,
                distance_range=None):
    raw = load_raw(raw_path)
    config = load_geometry()
    scale, bias = 1 + config["parameters"][4] / 100, config["parameters"][5]
    sensor = SensorModel()
    measured = sensor.predict(raw.voltage_V) if len(raw) else np.array([])
    # Use the voltage interval learned from the calibration CSV, not a fixed object distance.
    keep = sensor.measured_voltage(raw.voltage_V)
    outside_model = int((~keep).sum())
    if distance_range is not None:
        low, high = distance_range
        keep &= (measured * scale + bias >= low) & (measured * scale + bias <= high)
    cloud = raw.loc[keep, ["pan_deg", "tilt_deg", "voltage_V", "tilt_dir"]].reset_index(drop=True)
    pan = scan_axis(raw.pan_deg, PAN_STEP_DEG)
    tilt = scan_axis(raw.tilt_deg, TILT_STEP_DEG)
    rows, cols = np.searchsorted(tilt, cloud.tilt_deg), np.searchsorted(pan, cloud.pan_deg)
    grid = np.full((len(tilt), len(pan)), np.nan)
    grid[rows, cols] = measured[keep]
    volts = np.full(grid.shape, np.nan)
    volts[rows, cols] = cloud.voltage_V
    noise_v, noise_cm = noise(volts), noise(grid)
    measured = median_filter(grid, median)[rows, cols]
    samples = np.column_stack([cloud.pan_deg, cloud.tilt_deg, measured, cloud.tilt_dir])
    xyz = Geometry(samples, config).points() if len(cloud) else np.empty((0, 3))
    height, base = relief(xyz)
    cloud["distance_cm"] = measured * scale + bias
    cloud[["x_cm", "y_cm", "z_cm"]] = xyz
    cloud["height_cm"] = height
    indices = np.full(grid.shape, -1)
    indices[rows, cols] = np.arange(len(cloud))
    faces = mesh_faces(indices, xyz)
    grids = {name: np.full(grid.shape, np.nan) for name in GRIDS}
    for name, values in grids.items():
        values[rows, cols] = cloud[name]

    output.mkdir(parents=True, exist_ok=True)
    cloud.to_csv(output / "points.csv", index=False)
    np.savez(output / "depth.npz", pan_deg=pan, tilt_deg=tilt, valid=indices >= 0, **grids)
    write_ply(output / "surface.ply", xyz, faces)
    used = dict(config, reconstruction={"distance_range_cm": distance_range,
                                        "voltage_range_V": sensor.weights["voltage_range"].tolist(),
                                        "median_window": median,
                                        "noise_V": noise_v, "noise_cm": noise_cm})
    (output / "geometry_used.json").write_text(json.dumps(used, indent=2) + "\n")
    lag = config["parameters"][PARAMETER_NAMES.index("tilt_lag_deg")]
    vmin, vmax = sensor.weights["voltage_range"]
    print(f"Kept {len(cloud)}/{len(raw)} points; {len(faces)} triangles")
    print(f"Model voltage domain: {vmin:g}..{vmax:g} V; {outside_model} points outside training data")
    if distance_range is not None:
        print(f"Additional distance crop: {low:g}..{high:g} cm")
    print(f"Geometry: {'plane-calibrated' if config['calibrated'] else 'not plane-calibrated'}, "
          f"tilt lag {lag:+.2f} deg, median window {median}")
    if noise_cm is not None:
        print(f"Noise estimate before median: {noise_v:.3f} V = {noise_cm:.2f} cm per sample")
    if cloud.empty:
        placeholder(output, ["depth.png", "surface.png", "relief.png"],
                    "No points within the model domain and optional distance crop")
        print("No points in range; empty CSV/grid/PLY and explanatory images saved")
        return cloud
    if base is not None:
        print(f"Relief height 2..98 %: {np.percentile(height, 2):+.2f} .. "
              f"{np.percentile(height, 98):+.2f} cm above the base plane")
    plot_results(cloud, pan, tilt, grids, faces, base, output, show)
    print(f"Results: {output}")
    return cloud
