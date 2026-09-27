"""Read raw samples, keeping them available for later calibration/reconstruction."""

import csv
import time

import numpy as np
import pandas as pd
import serial

from settings import BAUD_RATE, RESET_WAIT_SECONDS, SERIAL_IDLE_SECONDS, TILT_CENTER

RAW_COLUMNS = ["pan_deg", "tilt_deg", "voltage_V"]


def tilt_direction(tilt, start=TILT_CENTER):
    """+1 / -1 when the tilt servo last moved toward larger / smaller angles.

    Needs acquisition order. Both firmwares start at the center and move up to TILT_MIN first.
    """
    step = np.sign(np.diff(np.asarray(tilt, dtype=float), prepend=start))
    return pd.Series(step).replace(0, np.nan).ffill().fillna(-1).to_numpy()


def load_raw(path):
    raw = pd.read_csv(path)
    raw["tilt_dir"] = tilt_direction(raw.tilt_deg)
    raw = raw.drop_duplicates(["pan_deg", "tilt_deg"], keep="last")
    if raw.empty:
        return raw
    return raw[np.isfinite(raw[RAW_COLUMNS]).all(axis=1)].reset_index(drop=True)


def read_scan(port):
    buffer = b""
    last_received = time.monotonic()
    while True:
        chunk = port.readline()
        if chunk:
            buffer += chunk
            last_received = time.monotonic()
        elif time.monotonic() - last_received > SERIAL_IDLE_SECONDS:
            raise TimeoutError("Arduino has stopped sending data")
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.decode("ascii").strip()
            if line == "SCAN_COMPLETE":
                return
            if line == "SCAN_ABORTED":
                raise RuntimeError("Arduino stopped the scan")
            if line.startswith("DATA,"):
                yield tuple(map(float, line.split(",")[1:]))


def request_stop(port):
    port.write(b"x")
    port.flush()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if b"MOTORS_OFF" in port.readline():
            print("Servos released")
            return
    print("Stop sent; no MOTORS_OFF reply received")


def acquire(port_name, output, command=b"s", columns=RAW_COLUMNS):
    output.parent.mkdir(parents=True, exist_ok=True)
    with serial.Serial(port_name, BAUD_RATE, timeout=0.5) as port, \
            output.open("w", newline="") as stream:
        time.sleep(RESET_WAIT_SECONDS)
        port.reset_input_buffer()
        writer = csv.writer(stream)
        writer.writerow(columns)
        port.write(command)
        try:
            for count, row in enumerate(read_scan(port), start=1):
                writer.writerow(row)
                stream.flush()
                if count % 50 == 0:
                    print(f"Received {count} samples")
        except BaseException:
            try:
                request_stop(port)
            except serial.SerialException:
                print("Serial disconnected; stop command could not be delivered")
            raise
    print(f"Raw samples saved: {output}")
    return pd.read_csv(output)
