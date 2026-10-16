"""Estimate the tilt servo lag from edges in any raw serpentine scan.

Neighboring pan columns are swept in opposite tilt directions. If the servo stops
short of each command by `lag` degrees, edges shift by 2 * lag between columns.
The lag that best aligns neighboring columns is the estimate. Raw voltage is used,
so the result does not depend on the distance model.
"""

import matplotlib.pyplot as plt
import numpy as np

from acquisition import load_raw
from common import save_figure
from geometry import TILT_LAG, load_geometry, save_geometry
from settings import GEOMETRY, OUTPUT

LAG_STEP_DEG = 0.05
LAGS_DEG = np.round(np.arange(-2, 2 + LAG_STEP_DEG / 2, LAG_STEP_DEG), 2)
# Both columns are smoothed alike; plain interpolation would favor half-sample shifts.
SMOOTH_DEG = 1.5
# A reliable minimum costs at least 5 % less than the lags 0.75 deg to either side.
CHECK_OFFSET = round(0.75 / LAG_STEP_DEG)
MIN_CONTRAST = 1.05


def columns(raw):
    """(tilt, direction, voltage) per complete pan column, sorted by tilt."""
    ordered = raw.sort_values("tilt_deg", kind="stable")
    cols = [(c.tilt_deg.to_numpy(), c.tilt_dir.to_numpy(), c.voltage_V.to_numpy())
            for _, c in ordered.groupby("pan_deg")]
    full = max(len(c[0]) for c in cols)
    return [c for c in cols if len(c[0]) == full]  # skip a partial column of an aborted scan


def smooth(at, tilt, value):
    weight = np.exp(-0.5 * ((at[:, None] - tilt[None, :]) / SMOOTH_DEG) ** 2)
    return weight @ value / weight.sum(axis=1)


def mismatch(cols, lag, at):
    """Mean |voltage difference| between neighboring columns at the same true tilt."""
    profiles = [smooth(at, t - d * lag, v) for t, d, v in cols]
    return float(np.mean([np.abs(a - b).mean() for a, b in zip(profiles[:-1], profiles[1:])]))


def estimate(raw):
    cols = columns(raw)
    edge = np.abs(LAGS_DEG).max()
    at = np.arange(cols[0][0].min() + edge, cols[0][0].max() - edge + 1e-9, 0.25)
    costs = np.array([mismatch(cols, lag, at) for lag in LAGS_DEG])
    best = int(costs.argmin())
    # Reliable only with a clear interior minimum, not at the edge of the search range.
    near = [costs[i] for i in (best - CHECK_OFFSET, best + CHECK_OFFSET) if 0 <= i < len(costs)]
    reliable = 0 < best < len(costs) - 1 and min(near) > MIN_CONTRAST * costs[best]
    return float(LAGS_DEG[best]), costs, reliable


def resample(raw, pan, tilt, lag):
    """Voltage image on commanded rows after moving each sample to its true tilt."""
    image = np.full((len(tilt), len(pan)), np.nan)
    for _, column in raw.groupby("pan_deg"):
        e = column.tilt_deg.to_numpy() - column.tilt_dir.to_numpy() * lag
        order = np.argsort(e, kind="stable")
        image[:, np.searchsorted(pan, column.pan_deg.iloc[0])] = np.interp(
            tilt, e[order], column.voltage_V.to_numpy()[order], left=np.nan, right=np.nan)
    return image


def plot(raw, lag, costs, show):
    pan, tilt = np.sort(raw.pan_deg.unique()), np.sort(raw.tilt_deg.unique())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    axes[0].plot(LAGS_DEG, costs)
    axes[0].axvline(lag, color="k", linestyle="--", label=f"best {lag:+.2f} deg")
    axes[0].set(xlabel="Tilt lag (deg)", ylabel="Neighbor column mismatch (V)",
                title="Lower = columns line up")
    axes[0].legend()
    for ax, value, title in ((axes[1], 0.0, "As commanded"),
                             (axes[2], lag, f"Lag {lag:+.2f} deg removed")):
        image = ax.pcolormesh(pan, tilt, resample(raw, pan, tilt, value),
                              shading="nearest", cmap="magma")
        ax.invert_yaxis()
        ax.set(xlabel="Pan command (deg)", ylabel="Tilt (deg)", title=title)
    fig.colorbar(image, ax=axes[1:], label="Sensor voltage (V), higher = closer")
    save_figure(fig, OUTPUT / "backlash.png", show)


def run(path, save=False, show=True):
    raw = load_raw(path)
    lag, costs, reliable = estimate(raw)
    zero = costs[np.flatnonzero(LAGS_DEG == 0)[0]]
    print(f"Tilt lag: {lag:+.2f} deg (total hysteresis {2 * abs(lag):.2f} deg)")
    print(f"Neighbor mismatch: {zero:.4f} V at 0 -> {costs.min():.4f} V")
    plot(raw, lag, costs, show)
    print(f"Saved {OUTPUT / 'backlash.png'}")
    if not reliable:
        print("Not reliable: the scan needs sharp edges crossing many columns. Nothing saved.")
    elif save:
        config = load_geometry(GEOMETRY)
        config["parameters"][TILT_LAG] = lag
        config["tilt_lag_source"] = str(path.resolve())
        save_geometry(config, GEOMETRY)
        print(f"Saved lag to {GEOMETRY}")
    return lag, reliable
