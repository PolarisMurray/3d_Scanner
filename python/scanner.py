"""Scan, capture calibration planes, calibrate, check, or rebuild a saved scan."""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from serial.tools import list_ports

from acquisition import acquire, request_stop, tilt_direction
from settings import BAUD_RATE, MEDIAN_WINDOW, OUTPUT, SERIAL_PORT


def odd_window(text):
    value = int(text)
    if value < 1 or value % 2 == 0:
        raise argparse.ArgumentTypeError("median window must be an odd number >= 1")
    return value


def add_processing(command):
    command.add_argument("--output", type=Path)
    command.add_argument("--median", type=odd_window, default=MEDIAN_WINDOW,
                         help="Odd pan-tilt window, e.g. 3; 1 = off")
    command.add_argument("--range", type=float, nargs=2, metavar=("MIN", "MAX"),
                         help="Optional extra distance crop in cm; default: model training domain")
    command.add_argument("--no-show", action="store_true")


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("ports", help="List serial ports")
    scan = commands.add_parser("scan", help="Acquire and reconstruct an object")
    scan.add_argument("--port", default=SERIAL_PORT)
    add_processing(scan)
    plane = commands.add_parser("plane", help="Acquire a known flat reference board")
    plane.add_argument("--port", default=SERIAL_PORT)
    plane.add_argument("--z", type=float, required=True,
                       help="Plane's intersection with +Z, measured from pan pivot, in cm")
    plane.add_argument("--normal", type=float, nargs=3, default=[0, 0, 1])
    plane.add_argument("--pan", type=float, nargs=2, default=[75, 105], metavar=("MIN", "MAX"))
    plane.add_argument("--tilt", type=float, nargs=2, default=[75, 90], metavar=("MIN", "MAX"))
    plane.add_argument("--output", type=Path, required=True)
    plane.add_argument("--from-raw", type=Path, help="Re-crop a saved raw board scan; no motion")
    calibration = commands.add_parser("calibrate", help="Optimize geometry with Taichi autodiff")
    calibration.add_argument("files", type=Path, nargs="+")
    calibration.add_argument("--fit-range", action="store_true",
                             help="Trust the entered board Z values and fit range scale/bias")
    calibration.add_argument("--no-show", action="store_true")
    check = commands.add_parser("check", help="Score flat boards that were not used for fitting")
    check.add_argument("files", type=Path, nargs="+")
    lag = commands.add_parser("backlash", help="Estimate tilt servo lag from a raw scan")
    lag.add_argument("csv", type=Path)
    lag.add_argument("--save", action="store_true", help="Store the lag in models/geometry.json")
    lag.add_argument("--no-show", action="store_true")
    rebuild = commands.add_parser("reconstruct", help="Rebuild a raw or original-project CSV")
    rebuild.add_argument("csv", type=Path)
    add_processing(rebuild)
    noise = commands.add_parser("noise", help="Hold still at 90/90 and measure sensor noise")
    noise.add_argument("--port", default=SERIAL_PORT)
    noise.add_argument("--output", type=Path)
    stop = commands.add_parser("stop", help="Send x when no other program owns the port")
    stop.add_argument("--port", default=SERIAL_PORT)
    return cli


def main():
    args = parser().parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if args.command == "ports":
        for port in list_ports.comports():
            print(f"{port.device}  {port.description}")
    elif args.command in ("scan", "reconstruct"):
        # Load the model before starting physical motion.
        from sensor import SensorModel
        from surface import reconstruct
        SensorModel()
        if args.command == "scan":
            output = args.output or OUTPUT / f"scan_{stamp}"
            raw = output / "raw.csv"
            acquire(args.port, raw)
        else:
            output = args.output or OUTPUT / f"rebuild_{stamp}"
            raw = args.csv
        reconstruct(raw, output, show=not args.no_show, median=args.median,
                    distance_range=args.range)
    elif args.command == "plane":
        from calibrate import PLANE_COLUMNS
        normal = np.array(args.normal, dtype=float)
        if not np.isfinite(normal).all() or np.linalg.norm(normal) == 0 or normal[2] == 0:
            raise ValueError("Plane normal must be finite, nonzero, with a nonzero Z component")
        normal /= np.linalg.norm(normal)
        if args.from_raw:
            import pandas as pd
            frame = pd.read_csv(args.from_raw)
        else:
            frame = acquire(args.port, args.output.parent / "raw" / args.output.name)
        # Direction needs the full acquisition order, so compute it before cropping.
        frame["tilt_dir"] = tilt_direction(frame.tilt_deg)
        frame = frame[frame.pan_deg.between(*args.pan) & frame.tilt_deg.between(*args.tilt)].copy()
        frame[PLANE_COLUMNS] = np.tile([*normal, normal[2] * args.z], (len(frame), 1))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"Saved {len(frame)} plane samples: {args.output}")
    elif args.command == "calibrate":
        from calibrate import calibrate
        calibrate(args.files, args.fit_range, show=not args.no_show)
    elif args.command == "check":
        from calibrate import check
        check(args.files)
    elif args.command == "backlash":
        import backlash
        backlash.run(args.csv, save=args.save, show=not args.no_show)
    elif args.command == "noise":
        import noise
        noise.run(args.port, args.output or OUTPUT / f"noise_{stamp}.csv")
    elif args.command == "stop":
        import serial
        import time
        from settings import RESET_WAIT_SECONDS
        with serial.Serial(args.port, BAUD_RATE, timeout=0.5) as port:
            time.sleep(RESET_WAIT_SECONDS)
            request_stop(port)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Scan interrupted; acquired raw samples kept")
