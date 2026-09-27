"""Stage 3: drive a fixed waypoint loop around the object, capture at each stop,
then hand the folder to the reconstruction pipeline.

Geometry (open loop, no feedback): aim the camera at the object once by turning the
turret sideways (--turret-offset-deg), then drive equal arcs of the given radius.
The tracks turn with the arc, so the camera stays pointed at the centre.

    python orbit_capture.py --dry-run --fake-camera --no-reconstruct     # no hardware at all
    python orbit_capture.py --dry-run --waypoints 20 --radius 0.6        # real camera, mock motion
    python orbit_capture.py --port /dev/cu.usbserial-0001 --waypoints 20 --radius 0.6
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from capture import Camera, CameraError, CaptureSession, FakeCamera
from control import ControlError, ExcavatorConfig, make_excavator
from control.api import ExcavatorController

log = logging.getLogger("orbit")


@dataclass(frozen=True)
class Waypoint:
    index: int
    angle_deg: float  # cumulative angle around the object (signed, + = CCW)
    step_deg: float  # arc to drive from the previous waypoint (0 for the first)


def plan_waypoints(n: int, arc_deg: float = 360.0) -> list[Waypoint]:
    """n waypoints evenly spaced over arc_deg. A full circle spaces them 360/n apart
    (the last one stops short of the start); a partial arc puts the first and last
    waypoints at its two ends."""
    if n < 2:
        raise ValueError("need at least 2 waypoints")
    step = arc_deg / n if abs(arc_deg) >= 360 else arc_deg / (n - 1)
    return [Waypoint(i, i * step, 0.0 if i == 0 else step) for i in range(n)]


def run_orbit(exc: ExcavatorController, camera, session: CaptureSession, *, radius: float,
              waypoints: list[Waypoint], turret_offset_deg: float, arm_position: float | None,
              settle_s: float, best_of: int, min_sharpness: float, retries: int) -> None:
    if arm_position is not None:
        exc.home_arm()
        exc.move_arm(arm_position)
    if turret_offset_deg:
        exc.rotate_turret(turret_offset_deg)

    for wp in waypoints:
        if wp.step_deg:
            exc.drive_arc(radius, wp.step_deg)
        for attempt in range(retries + 1):
            time.sleep(settle_s * (attempt + 1))  # let vibration die down (longer on retries)
            frame, sharp = camera.grab_best(best_of)
            if sharp >= min_sharpness:
                break
            log.warning("waypoint %d: soft frame (sharpness %.0f < %.0f)%s", wp.index, sharp,
                        min_sharpness, ", retaking" if attempt < retries else ", keeping it")
        path = session.save_frame(frame, waypoint=wp.index,
                                  commanded_angle_deg=round(wp.angle_deg, 2),
                                  sharpness=round(sharp, 1), source="orbit")
        log.info("waypoint %2d/%d  angle %6.1f  -> %s", wp.index + 1, len(waypoints),
                 wp.angle_deg, path.name)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("motion")
    g.add_argument("--waypoints", type=int, default=20)
    g.add_argument("--radius", type=float, required=True,
                   help="arc radius in metres (distance from the object to the excavator centre)")
    g.add_argument("--arc-deg", type=float, default=360.0, help="total angle to sweep (default full circle)")
    g.add_argument("--direction", choices=("ccw", "cw"), default="ccw")
    g.add_argument("--turret-offset-deg", type=float, default=90.0,
                   help="turn the turret this far toward the object side before orbiting (0 = don't)")
    g.add_argument("--arm-position", type=float, default=None,
                   help="home the arm, then move it to this 0-1 position first")
    g.add_argument("--settle-s", type=float, default=1.0, help="wait after each move before shooting")
    g.add_argument("--config", help="control JSON (calibration + axis mapping)")
    g.add_argument("--port", default=os.environ.get("EXCAVATOR_PORT"))
    g.add_argument("--dry-run", action="store_true", help="mock the excavator (print commands)")
    g.add_argument("-y", "--yes", action="store_true", help="don't ask before moving the machine")

    c = p.add_argument_group("camera / session")
    c.add_argument("--camera", type=int, default=0)
    c.add_argument("--fake-camera", action="store_true", help="synthetic frames, for testing only")
    c.add_argument("--best-of", type=int, default=5, help="frames grabbed per waypoint; sharpest kept")
    c.add_argument("--min-sharpness", type=float, default=60.0)
    c.add_argument("--retries", type=int, default=2, help="retakes for soft frames")
    c.add_argument("--session-root", default="captures")
    c.add_argument("--session-name", default=None)

    r = p.add_argument_group("reconstruction")
    r.add_argument("--no-reconstruct", action="store_true", help="stop after capturing")
    r.add_argument("--colmap-bin", default=os.environ.get("COLMAP_BIN", "colmap"))
    r.add_argument("--cpu-only", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    sign = 1.0 if args.direction == "ccw" else -1.0
    waypoints = plan_waypoints(args.waypoints, args.arc_deg * sign)
    session = CaptureSession(args.session_root, args.session_name)
    settle = 0.0 if args.dry_run else args.settle_s

    try:
        cfg = ExcavatorConfig.from_file(args.config) if args.config else None
        camera = FakeCamera() if args.fake_camera else Camera(args.camera)
        with camera, make_excavator(args.port, args.dry_run, cfg) as exc:
            session.write_info({"args": vars(args), "waypoints": [w.__dict__ for w in waypoints]})
            try:
                if not args.dry_run and not args.yes:
                    input(f"Excavator will move ({len(waypoints)} waypoints, radius {args.radius} m). "
                          "Clear the area, then press Enter (Ctrl-C aborts)... ")
                run_orbit(exc, camera, session, radius=args.radius, waypoints=waypoints,
                          turret_offset_deg=args.turret_offset_deg * sign,
                          arm_position=args.arm_position, settle_s=settle,
                          best_of=args.best_of, min_sharpness=args.min_sharpness,
                          retries=args.retries)
            except KeyboardInterrupt:
                log.warning("Interrupted; machine stopped. %d frame(s) kept.", session.count)
                return 130
            if args.dry_run:
                log.info("(dry-run: mock motion would take about %.0f s)", exc.dry_run_elapsed_s)
    except (ControlError, CameraError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    log.info("Captured %d frame(s) in %s", session.count, session.images_dir)
    if args.no_reconstruct:
        return 0

    from reconstruct import ReconstructionConfig, ReconstructionError, run_reconstruction
    try:
        result = run_reconstruction(ReconstructionConfig(
            image_dir=session.images_dir, colmap_bin=args.colmap_bin, cpu_only=args.cpu_only))
    except ReconstructionError as e:
        print(f"\nReconstruction failed (your {session.count} images are safe in "
              f"{session.images_dir}):\n{e}", file=sys.stderr)
        return 3
    if result.mesh:
        log.info("Mesh: %s", result.mesh.stl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
