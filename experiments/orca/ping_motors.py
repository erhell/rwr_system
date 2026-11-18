#!/usr/bin/env python
"""Read-only check of the ORCA hand's 17 Feetech HLS motors: do they answer, at what supply voltage and temperature,
with which error flags. Nothing is written: no torque, no motion, no register change.

Written 2026-11-16 after all motors stopped answering mid-run (feedback froze, the e-stop could not even disable
torque): run it to see when the motors are back after a power cycle, and whether one is hot or reports an error.

Usage (with the hand controller stopped, it holds the port):
    python ping_motors.py            # one check
    python ping_motors.py --watch    # repeat every 2 s until Ctrl+C

Registers (HLS memory table, orca_core feetech_registers.py): 62 present voltage (0.1 V), 63 present temperature (C),
65 status (bit 0 voltage, 1 sensor, 2 temperature, 3 current, 4 angle, 5 overload), 40 torque enable, 69 present
current (6.5 mA, bit 15 sign).
"""

import argparse
import glob
import sys
import time

from orca_core.hardware.feetech import COMM_SUCCESS, PortHandler
from orca_core.hardware.feetech_client import HLSPacketHandler

IDS = range(1, 18)
NAMES = {1: "wrist"}  # the others by ID; joint_to_motor_map is in the hand config
BAUD = 1_000_000
VOLTAGE, TEMPERATURE, STATUS, TORQUE_ENABLE, PRESENT_CURRENT = 62, 63, 65, 40, 69
FLAGS = ["voltage", "sensor", "temperature", "current", "angle", "overload"]


def find_port() -> str | None:
    """The serial port on which motor 1 or 17 answers a ping."""
    for port in sorted(glob.glob("/dev/serial/by-id/*ORCA*")) + sorted(glob.glob("/dev/ttyACM*")):
        ph = PortHandler(port)
        ph.baudrate = BAUD
        try:
            if not ph.openPort():
                continue
            pk = HLSPacketHandler(ph)
            for mid in (1, 17):
                if pk.ping(mid)[1] == COMM_SUCCESS:
                    return port
        except Exception:
            pass
        finally:
            try:
                ph.closePort()
            except Exception:
                pass
    return None


def check(port: str) -> int:
    ph = PortHandler(port)
    ph.baudrate = BAUD
    try:
        opened = ph.openPort()
    except Exception as exc:  # unplugged: the port vanished
        opened = False
        print(f"cannot open {port}: {exc}")
    if not opened:
        print(f"cannot open {port} (hand unplugged, or the hand controller still holds it)")
        return 0
    pk = HLSPacketHandler(ph)
    ok = 0
    try:
        print(f"{'ID':>3} {'motor':8} {'answers':8} {'volt':>6} {'temp':>5} {'torque':>6} {'current':>8}  flags")
        for mid in IDS:
            _, res, _ = pk.ping(mid)
            if res != COMM_SUCCESS:
                print(f"{mid:3d} {NAMES.get(mid, ''):8} {'NO':8}")
                continue
            ok += 1
            v, r1, _ = pk.read1ByteTxRx(mid, VOLTAGE)
            t, r2, _ = pk.read1ByteTxRx(mid, TEMPERATURE)
            s, r3, _ = pk.read1ByteTxRx(mid, STATUS)
            te, r4, _ = pk.read1ByteTxRx(mid, TORQUE_ENABLE)
            c, r5, _ = pk.read2ByteTxRx(mid, PRESENT_CURRENT)
            cur = (-(c & 0x7FFF) if c & 0x8000 else c) * 6.5 if r5 == COMM_SUCCESS else float("nan")
            flags = [f for i, f in enumerate(FLAGS) if r3 == COMM_SUCCESS and s >> i & 1]
            print(f"{mid:3d} {NAMES.get(mid, ''):8} {'yes':8} {v / 10 if r1 == COMM_SUCCESS else float('nan'):5.1f}V"
                  f" {t if r2 == COMM_SUCCESS else -1:4d}C {('on' if te else 'off') if r4 == COMM_SUCCESS else '?':>6}"
                  f" {cur:6.0f}mA  {', '.join(flags) if flags else '-'}")
    finally:
        ph.closePort()
    print(f"{ok} of {len(IDS)} motors answer on {port}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--watch", action="store_true", help="repeat every 2 s until Ctrl+C")
    ap.add_argument("--port", default=None, help="serial port (default: the one where the motors answer)")
    args = ap.parse_args()
    while True:
        port = args.port or find_port()
        if port is None:
            print(f"{time.strftime('%H:%M:%S')} no motor answers on any ORCA / ttyACM port: motor power off or bus "
                  "disconnected (the USB board itself: " + (", ".join(glob.glob('/dev/ttyACM*')) or "not found") + ")")
        else:
            print(time.strftime("%H:%M:%S"))
            n = check(port)
            if args.watch and n == len(IDS):
                print("all motors are back")
                return 0
        if not args.watch:
            return 0
        time.sleep(2)


if __name__ == "__main__":
    sys.exit(main())
