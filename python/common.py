"""Small helpers shared by the processing and calibration scripts."""

import json

import matplotlib.pyplot as plt
import numpy as np

# For Gaussian noise, median absolute deviation * 1.4826 equals the standard deviation,
# but a few outliers barely move it.
MAD_TO_SIGMA = 1.4826


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def robust_sigma(values, center=0.0):
    """Outlier-resistant standard deviation of `values` around `center`."""
    return float(MAD_TO_SIGMA * np.median(np.abs(np.asarray(values) - center)))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def save_figure(fig, path, show, dpi=160):
    """Save, optionally display, then free the figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
