"""Developer checks only; no hardware, no fabricated measurement files."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import acquisition
import backlash
import calibrate
from calibrate import fit_geometry
from geometry import Geometry, fit_plane
from settings import default_geometry
from sensor import SensorModel
from surface import median_filter, mesh_faces, reconstruct, relief


def reference_point(pan, tilt, distance, direction, parameters, config):
    """Independent NumPy rotation matrices for cross-checking Taichi."""
    tilt = tilt - direction * parameters[6]
    theta = np.deg2rad((pan - 90) * (1 + parameters[2] / 100) + parameters[0])
    phi = np.deg2rad((tilt - 90) * (1 + parameters[3] / 100) + parameters[1])
    ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    rx = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi), np.cos(phi)]])
    d = distance * (1 + parameters[4] / 100) + parameters[5]
    return ry @ (config["pan_to_tilt_cm"] +
                 rx @ (np.array(config["tilt_to_sensor_cm"]) + [0, 0, d]))


def board_scans(truth, config, z_error=(0, 0, 0)):
    """Exact samples of three boards; z_error = mistakes in the entered board Z."""
    samples, planes, boards = [], [], []
    for k, (normal, z) in enumerate([([0, 0, 1], 40), ([0.3, 0, 1], 50), ([0, -0.3, 1], 60)]):
        n = np.array(normal) / np.linalg.norm(normal)
        for column, pan in enumerate(np.linspace(60, 120, 9)):
            for tilt in np.linspace(62, 95, 7):
                direction = 1 if column % 2 == 0 else -1
                origin = reference_point(pan, tilt, 0, direction, truth, config)
                ray = reference_point(pan, tilt, 1, direction, truth, config) - origin
                samples.append([pan, tilt, (n[2] * z - n @ origin) / (n @ ray), direction])
                planes.append([*n, n[2] * (z + z_error[k])])
                boards.append(k)
    return np.array(samples), np.array(planes), np.array(boards)


def serpentine(pans, tilts):
    order = []
    for column, pan in enumerate(pans):
        order += [(pan, tilt) for tilt in (tilts if column % 2 == 0 else tilts[::-1])]
    return pd.DataFrame(order, columns=["pan_deg", "tilt_deg"])


class PipelineTests(unittest.TestCase):
    def test_unobservable_parameter_has_unknown_uncertainty(self):
        class SimpleEngine:
            def points(self, x):
                return np.column_stack([np.zeros(10), np.zeros(10),
                                         50 + np.linspace(1, 2, 10) * x[0]])
        x = np.zeros(8)
        x[0] = 0.5
        planes = np.tile([0, 0, 1, 50.0], (10, 1))
        std, _ = calibrate.spread(SimpleEngine(), x, [0, 2], planes, np.zeros(10, int))
        self.assertTrue(np.isfinite(std[0]))
        self.assertTrue(np.isinf(std[1]))
        std, correlation = calibrate.spread(SimpleEngine(), x, [], planes, np.zeros(10, int))
        self.assertEqual(std.size, 0)
        self.assertEqual(correlation.shape, (0, 0))

    def test_kinematics_with_offsets_and_lag(self):
        config = default_geometry()
        config.update(pan_to_tilt_cm=[1, 3, 2], tilt_to_sensor_cm=[0.5, -1, 4],
                      parameters=[2, -3, 4, -5, 1, 0.4, 0.7])
        samples = np.array([[90, 90, 50, 1], [55, 60, 45, -1], [120, 95, 65, 0]])
        expected = [reference_point(*row, config["parameters"], config) for row in samples]
        np.testing.assert_allclose(Geometry(samples, config).points(), expected, atol=1e-10)

    def test_autodiff_matches_finite_difference(self):
        samples = [[75, 82, 48, 1], [97, 93, 52, -1], [112, 74, 60, 1]]
        engine = Geometry(samples, default_geometry(), boards=[0, 1, 1])
        engine.planes.from_numpy(np.array([[0, 0, 1, 50]] * 3, dtype=float))
        x = np.array([2, -1, 3, -2, 1, 0.2, 0.5, 0.3, -0.4])
        _, gradient = engine.objective(x)
        numerical = []
        for i in range(len(x)):
            step = np.zeros(len(x))
            step[i] = 1e-4
            numerical.append((engine.objective(x + step)[0] - engine.objective(x - step)[0]) / 2e-4)
        np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-6)

    def test_measured_boards_recover_angles_and_range(self):
        config = default_geometry()
        config.update(pan_to_tilt_cm=[0.5, 2, 1], tilt_to_sensor_cm=[0, 0.5, 3])
        truth = np.array([2, -1.5, 3, -2, 1.2, 0.4, 0.6])
        config["parameters"] = [0, 0, 0, 0, 0, 0, truth[6]]
        samples, planes, boards = board_scans(truth, config)
        fit = fit_geometry(samples, planes, boards, config, True)
        self.assertTrue(fit.result.success, fit.result.message)
        self.assertLess(np.sqrt(np.mean(fit.after**2)), 0.01)
        self.assertLess(np.linalg.norm(fit.after), np.linalg.norm(fit.before) / 50)
        np.testing.assert_allclose(fit.result.x[:7], truth, atol=0.06)
        self.assertEqual((fit.locked, fit.at_limit), ({}, []))

    def test_fitted_board_distance_tolerates_tape_errors(self):
        config = default_geometry()
        config.update(pan_to_tilt_cm=[0.5, 2, 1], tilt_to_sensor_cm=[0, 0.5, 3])
        truth = np.array([2, -1.5, 3, -2, 0, 0, 0.6])
        config["parameters"] = [0, 0, 0, 0, 0, 0, truth[6]]
        samples, planes, boards = board_scans(truth, config, z_error=(0.8, -0.5, 0.3))
        fit = fit_geometry(samples, planes, boards, config)
        self.assertTrue(fit.result.success, fit.result.message)
        self.assertLess(np.sqrt(np.mean(fit.after**2)), 0.01)
        np.testing.assert_allclose(fit.result.x[:7], truth, atol=0.06)
        entered_z = planes[[0, 63, 126], 3] / planes[[0, 63, 126], 2]
        fitted_z = entered_z + fit.result.x[7:] / planes[[0, 63, 126], 2]
        np.testing.assert_allclose(fitted_z, [40, 50, 60], atol=0.05)
        self.assertEqual(fit.free, [0, 1, 2, 3, 7, 8, 9])
        self.assertTrue(np.isfinite(fit.std).all())

    def test_weakly_determined_corrections_stay_zero(self):
        config = default_geometry()
        truth = np.array([1.5, -2, 3, -4, 0, 0, 0])
        samples, planes, boards = board_scans(truth, config)
        one_board = boards == 0
        noisy = samples[one_board].copy()
        noisy[:, 2] += np.random.default_rng(2).normal(0, 0.5, len(noisy))
        fit = fit_geometry(noisy, planes[one_board], boards[one_board], config)
        self.assertTrue(fit.locked)
        for i in fit.locked:
            self.assertEqual(fit.result.x[i], 0)
        for i in fit.free:
            if i < 4:
                self.assertLessEqual(fit.std[fit.free.index(i)], calibrate.USEFUL_SIGMA[i])

    def test_calibrate_and_check_keep_lag_and_write_results(self):
        config = default_geometry()
        truth = np.array([1.5, -1, 2, -2, 0, 0, 0.5])
        config["parameters"] = [0, 0, 0, 0, 0, 0, truth[6]]
        samples, planes, boards = board_scans(truth, config, z_error=(0.6, -0.4, 0.2))
        # Identity sensor: the voltage column already holds the distance.
        sensor = unittest.mock.Mock()
        sensor.return_value.predict.side_effect = lambda v: np.asarray(v, dtype=float)
        sensor.return_value.measured_voltage.side_effect = lambda v: np.ones(len(v), bool)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            paths = []
            for k in range(3):
                frame = pd.DataFrame(samples[boards == k],
                                     columns=["pan_deg", "tilt_deg", "voltage_V", "tilt_dir"])
                frame[calibrate.PLANE_COLUMNS] = planes[boards == k]
                paths.append(root / f"board{k}.csv")
                frame.to_csv(paths[-1], index=False)
            geometry = root / "geometry.json"
            geometry.write_text(json.dumps(config))
            with patch.object(calibrate, "SensorModel", sensor), \
                    patch.object(calibrate, "GEOMETRY", geometry), \
                    patch.object(calibrate, "OUTPUT", root):
                calibrate.calibrate(paths, show=False)
                calibrate.check(paths[:1])
                saved = json.loads(geometry.read_text())
                self.assertTrue((root / "geometry_calibration.png").exists())
                # Scan-level noise (several cm): nothing trustworthy, so nothing is saved.
                for path in paths:
                    frame = pd.read_csv(path)
                    frame["voltage_V"] += np.random.default_rng(3).normal(0, 4, len(frame))
                    frame.to_csv(path, index=False)
                try:
                    calibrate.calibrate(paths, show=False)
                except RuntimeError:
                    pass
                self.assertEqual(json.loads(geometry.read_text()), saved)
        np.testing.assert_allclose(saved["parameters"], truth, atol=0.06)
        self.assertEqual(saved["fit_mode"], "fitted board Z")

    def test_tilt_direction_follows_serpentine_order(self):
        tilt = [60, 61.5, 63, 63, 61.5, 60, 60, 61.5]
        np.testing.assert_array_equal(acquisition.tilt_direction(tilt),
                                      [-1, 1, 1, 1, -1, -1, -1, 1])

    def test_backlash_estimate_recovers_lag(self):
        raw = serpentine(np.arange(55, 120.1, 1.5), np.arange(60, 95.1, 1.5))
        raw["tilt_dir"] = acquisition.tilt_direction(raw.tilt_deg)
        true_tilt = raw.tilt_deg - raw.tilt_dir * 0.6
        slanted = true_tilt - 0.05 * (raw.pan_deg - 90)
        edge = lambda at: 1 / (1 + np.exp(-(slanted - at) / 0.4))
        raw["voltage_V"] = 0.8 + 1.2 * (edge(70.3) - edge(80.7) + edge(85.2) - edge(88.9))
        lag, _, reliable = backlash.estimate(raw)
        self.assertTrue(reliable)
        self.assertAlmostEqual(lag, 0.6, delta=0.1)
        raw["voltage_V"] = 1.2 + np.random.default_rng(0).normal(0, 0.01, len(raw))
        self.assertFalse(backlash.estimate(raw)[2])

    def test_median_filter_removes_spike_but_not_holes(self):
        grid = np.full((5, 5), 50.0)
        grid[2, 2] = 60
        grid[0, 4] = np.nan
        filtered = median_filter(grid, 3)
        self.assertEqual(filtered[2, 2], 50)
        self.assertTrue(np.isnan(filtered[0, 4]))
        np.testing.assert_array_equal(filtered[np.isfinite(filtered)], 50)
        self.assertIs(median_filter(grid, 1), grid)
        # Two surfaces meet: every output is a measured value, never a blend.
        step = np.array([[40.0, 40, 65, 65], [40, 40, 65, 65], [40, 40, 65, 65]])
        self.assertTrue(np.isin(median_filter(step, 3), [40, 65]).all())
        sparse = np.array([[40.0, 65, np.nan], [np.nan, np.nan, np.nan]])
        np.testing.assert_array_equal(median_filter(sparse, 3), sparse)

    def test_relief_height_on_tilted_board(self):
        rng = np.random.default_rng(1)
        x, y = rng.uniform(-15, 15, (2, 500))
        xyz = np.column_stack([x, y, 50 + 0.2 * x - 0.1 * y])
        toward_scanner = np.array([0.2, -0.1, -1]) / np.linalg.norm([0.2, -0.1, -1])
        raised = x**2 + y**2 < 30
        xyz += 1.5 * raised[:, None] * toward_scanner + rng.normal(0, 0.05, xyz.shape)
        height, (center, normal) = relief(xyz)
        self.assertAlmostEqual(np.median(height[raised]), 1.5, delta=0.1)
        self.assertLess(abs(np.median(height[~raised])), 0.05)
        self.assertLess(normal[2], 0)

    def test_plane_fit_keeps_facing_scanner_on_small_noisy_patch(self):
        rng = np.random.default_rng(4)
        x, y = rng.uniform(-6.5, 6.5, 220), rng.uniform(-3.5, 3.5, 220)
        center, normal, _ = fit_plane(np.column_stack([x, y, 25 + rng.normal(0, 3, 220)]))
        self.assertLess(np.degrees(np.arccos(-normal[2])), 20)
        self.assertAlmostEqual(center[2], 25, delta=1)

    def test_missing_column_is_not_bridged(self):
        xyz = np.array([[0, 0, 50], [3, 0, 50], [0, 1.5, 50], [3, 1.5, 50], [1.5, 0, 50]])
        self.assertEqual(len(mesh_faces(np.array([[0, -1, 1], [2, -1, 3]]), xyz)), 0)
        cell = np.array([[0, 4], [2, 3]])
        self.assertEqual(len(mesh_faces(cell, xyz)), 2)
        self.assertEqual(len(mesh_faces(cell, xyz, edge_limit=2.5)), 1)

    def test_reconstruct_uses_acquisition_order_for_lag(self):
        config = default_geometry()
        config["parameters"][6] = 1.0
        raw = serpentine([88.5, 90], [60, 61.5, 63])
        raw["voltage_V"] = 1.0
        with tempfile.TemporaryDirectory() as name, \
                patch("surface.SensorModel") as model, patch("surface.plot_results"), \
                patch("surface.load_geometry", return_value=config):
            path = Path(name) / "raw.csv"
            raw.to_csv(path, index=False)
            model.return_value.weights = {"voltage_range": np.array([0, 5])}
            model.return_value.measured_voltage.side_effect = lambda v: np.ones(len(v), bool)
            model.return_value.predict.return_value = np.full(len(raw), 50.0)
            cloud = reconstruct(path, Path(name) / "out", False)
        expected = [reference_point(p, t, 50, d, config["parameters"], config)
                    for p, t, d in zip(cloud.pan_deg, cloud.tilt_deg, [-1, 1, 1, 1, -1, -1])]
        np.testing.assert_allclose(cloud[["x_cm", "y_cm", "z_cm"]], expected, atol=1e-9)

    def test_model_domain_replaces_fixed_distance_crop(self):
        sensor = SensorModel()
        low, high = sensor.weights["voltage_range"]
        with tempfile.TemporaryDirectory() as name, patch("surface.plot_results"), \
                patch("surface.load_geometry", return_value=default_geometry()):
            root = Path(name)
            raw = serpentine([88.5, 90], [90, 91.5])
            raw["voltage_V"] = [low - 0.01, low, high, high + 0.01]
            raw.to_csv(root / "raw.csv", index=False)
            cloud = reconstruct(root / "raw.csv", root / "out", False, median=1)
            np.testing.assert_allclose(cloud.voltage_V, [low, high])
            np.testing.assert_allclose(cloud.distance_cm, sensor.predict([low, high]))
            used = json.loads((root / "out/geometry_used.json").read_text())["reconstruction"]
            self.assertIsNone(used["distance_range_cm"])
            self.assertEqual(used["voltage_range_V"], [low, high])

    def test_optional_range_boundaries_and_empty_scan(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "raw.csv"
            pd.DataFrame([[90, 90, 1], [91.5, 90, 2], [93, 90, 3], [94.5, 90, 4]],
                         columns=acquisition.RAW_COLUMNS).to_csv(path, index=False)
            with patch("surface.SensorModel") as model, patch("surface.plot_results"), \
                    patch("surface.load_geometry", return_value=default_geometry()):
                model.return_value.weights = {"voltage_range": np.array([0, 5])}
                model.return_value.measured_voltage.side_effect = lambda v: np.ones(len(v), bool)
                model.return_value.predict.return_value = np.array([39.9, 40, 65, 65.1])
                cloud = reconstruct(path, root / "cloud", False, distance_range=(40, 65))
                self.assertEqual(cloud.distance_cm.tolist(), [40, 65])
                model.return_value.predict.return_value = np.array([10, 20, 70, 100])
                cloud = reconstruct(path, root / "empty", False, distance_range=(40, 65))
                self.assertTrue(cloud.empty)
                self.assertTrue((root / "empty/surface.ply").exists())
                self.assertTrue((root / "empty/relief.png").exists())

    def test_header_only_scan_and_removed_column(self):
        with tempfile.TemporaryDirectory() as name, \
                patch("surface.SensorModel") as model, patch("surface.plot_results"), \
                patch("surface.load_geometry", return_value=default_geometry()):
            root = Path(name)
            path = root / "raw.csv"
            model.return_value.weights = {"voltage_range": np.array([0, 5])}
            model.return_value.measured_voltage.side_effect = lambda v: np.ones(len(v), bool)
            pd.DataFrame(columns=acquisition.RAW_COLUMNS).to_csv(path, index=False)
            self.assertTrue(reconstruct(path, root / "empty", False).empty)
            raw = serpentine([88.5, 91.5], [90, 91.5])
            raw["voltage_V"] = 1.0
            raw.to_csv(path, index=False)
            model.return_value.predict.return_value = np.full(4, 50.0)
            reconstruct(path, root / "gap", False)
            with np.load(root / "gap/depth.npz") as grid:
                np.testing.assert_array_equal(grid["pan_deg"], [88.5, 90, 91.5])
                self.assertFalse(grid["valid"][:, 1].any())
            self.assertIn("element face 0", (root / "gap/surface.ply").read_text())

    def test_serial_fragmentation_and_abort(self):
        chunks = iter([b"READY\n", b"DATA,88.5,91", b".5,1.23\r\n", b"SCAN_COMPLETE\n"])
        port = unittest.mock.Mock()
        port.readline.side_effect = lambda: next(chunks)
        self.assertEqual(list(acquisition.read_scan(port)), [(88.5, 91.5, 1.23)])
        port.readline.side_effect = [b"SCAN_ABORTED\n"]
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            list(acquisition.read_scan(port))

    def test_stop_waits_for_release(self):
        port = unittest.mock.Mock()
        port.readline.side_effect = [b"SCAN_ABORTED\n", b"MOTORS_OFF\n"]
        acquisition.request_stop(port)
        port.write.assert_called_once_with(b"x")
        port.flush.assert_called_once()

    def test_serial_idle_times_out(self):
        port = unittest.mock.Mock()
        port.readline.return_value = b""
        with patch.object(acquisition.time, "monotonic", side_effect=[0, 20]):
            with self.assertRaises(TimeoutError):
                list(acquisition.read_scan(port))

    def test_ctrl_c_sends_stop_and_keeps_raw(self):
        port = unittest.mock.MagicMock()
        port.__enter__.return_value = port
        with tempfile.TemporaryDirectory() as name, \
                patch.object(acquisition.serial, "Serial", return_value=port), \
                patch.object(acquisition.time, "sleep"), \
                patch.object(acquisition, "request_stop") as stop:
            path = Path(name) / "raw.csv"
            def interrupted(_):
                yield (90, 90, 1.5)
                raise KeyboardInterrupt
            with patch.object(acquisition, "read_scan", interrupted), self.assertRaises(KeyboardInterrupt):
                acquisition.acquire("fake", path)
            stop.assert_called_once_with(port)
            self.assertEqual(len(pd.read_csv(path)), 1)


if __name__ == "__main__":
    unittest.main()
