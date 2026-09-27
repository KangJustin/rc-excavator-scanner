"""Hardware bring-up / manual control CLI.

    python -m control ports
    python -m control --port /dev/cu.usbserial-0001 ping
    python -m control --port ... set-stick left_x 200 --hold 1
    python -m control --port ... set-line W 1 --hold 1
    python -m control --port ... press boom_up 300
    python -m control --dry-run drive-arc 0.5 90
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .api import make_excavator
from .config import ExcavatorConfig
from .transport import ControlError


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m control", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default=os.environ.get("EXCAVATOR_PORT"),
                   help="serial port (or set EXCAVATOR_PORT)")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--dry-run", action="store_true", help="print commands instead of sending them")
    p.add_argument("--config", help="JSON calibration/axis-mapping file")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ports", help="list serial ports")
    sub.add_parser("print-config", help="print the default config as JSON (edit + pass via --config)")
    sub.add_parser("ping")
    sub.add_parser("neutral", help="centre all sticks, release all buttons")
    s = sub.add_parser("set-stick"); s.add_argument("name"); s.add_argument("value", type=int)
    s.add_argument("--hold", type=float, default=1.0, help="seconds to hold before returning to neutral (default 1)")
    s = sub.add_parser("set-line", help="drive a digital line (see control.config.LINE_NAMES) high (1) or low (0)")
    s.add_argument("name"); s.add_argument("level", type=int, choices=(0, 1))
    s.add_argument("--hold", type=float, default=1.0, help="seconds to hold before returning to neutral (default 1)")
    s = sub.add_parser("press"); s.add_argument("name"); s.add_argument("duration_ms", type=float)
    s = sub.add_parser("drive-arc"); s.add_argument("radius", type=float); s.add_argument("angle_deg", type=float)
    s = sub.add_parser("turret"); s.add_argument("angle_deg", type=float)
    s = sub.add_parser("arm"); s.add_argument("position", type=float)
    s.add_argument("--current", type=float,
                   help="where the arm is now (0-1); omitted = home to the 0.0 stop first")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.cmd == "ports":
        from serial.tools import list_ports
        for port in list_ports.comports():
            print(f"{port.device}\t{port.description}")
        return 0
    if args.cmd == "print-config":
        print(json.dumps(ExcavatorConfig().to_dict(), indent=2))
        return 0

    try:
        cfg = ExcavatorConfig.from_file(args.config) if args.config else None
        with make_excavator(args.port, args.dry_run, cfg, args.baud) as exc:
            if args.cmd == "ping":
                print("ESP32 reachable." if not args.dry_run else "(dry-run)")
            elif args.cmd == "neutral":
                exc.stop()
            elif args.cmd == "set-stick":
                exc.set_stick(args.name, args.value)
                exc.hold(args.hold)
            elif args.cmd == "set-line":
                exc.set_line(args.name, bool(args.level))
                exc.hold(args.hold)
            elif args.cmd == "press":
                exc.press_button(args.name, args.duration_ms)
            elif args.cmd == "drive-arc":
                exc.drive_arc(args.radius, args.angle_deg)
            elif args.cmd == "turret":
                exc.rotate_turret(args.angle_deg)
            elif args.cmd == "arm":
                if args.current is None:
                    exc.home_arm()
                else:
                    exc.set_arm_estimate(args.current)
                exc.move_arm(args.position)
            if args.dry_run:
                print(f"(dry-run: would take {exc.dry_run_elapsed_s:.1f} s)")
    except (ControlError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
