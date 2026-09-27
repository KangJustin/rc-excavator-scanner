"""Capture stills from the webcam for 3D reconstruction.

    python -m capture                      # live preview
    python -m capture --check              # is the feed clean at my working distance?
    python -m capture --list-cameras       # saves a snapshot per camera so you can tell them apart
    python -m capture --camera-index 1     # pick a camera (aliases: --camera, --device)

Preview hotkeys:   SPACE = take photo    N = start a new session folder    Q / ESC = quit
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Callable

import cv2
import numpy as np

from .camera import MIN_BRIGHTNESS, Camera, CameraError, frame_stats, light_status, list_cameras, measure_fps, sharpness
from .session import CaptureSession

WIN = "capture"
TARGET_MIN, TARGET_MAX = 20, 30  # recommended number of photos per orbit
TOAST_SECONDS = 1.5


def draw_overlay(frame: np.ndarray, session_name: str, count: int, sharp: float,
                 min_sharpness: float, toast: str | None = None, camera_label: str = "") -> np.ndarray:
    """Big running count for whoever is holding the controller, plus session name,
    sharpness warning, last-action message and hotkey reminder."""
    out = frame.copy()
    h, w = out.shape[:2]
    bar = out.copy()
    cv2.rectangle(bar, (0, 0), (w, 92), (0, 0, 0), -1)
    cv2.rectangle(bar, (0, h - 34), (w, h), (0, 0, 0), -1)
    out = cv2.addWeighted(bar, 0.55, out, 0.45, 0)

    if count < TARGET_MIN:
        color = (255, 255, 255)  # keep going
    elif count <= TARGET_MAX:
        color = (0, 220, 0)  # in the sweet spot
    else:
        color = (0, 200, 255)  # plenty already
    cv2.putText(out, f"SHOTS: {count}", (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4, cv2.LINE_AA)
    cv2.putText(out, f"session {session_name}   (aim for {TARGET_MIN}-{TARGET_MAX})", (w // 2 - 60, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    soft = sharp < min_sharpness
    cv2.putText(out, f"sharpness {sharp:.0f}" + ("  SOFT - hold still" if soft else ""),
                (w // 2 - 60, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if soft else (0, 220, 0), 2, cv2.LINE_AA)
    if camera_label:
        cv2.putText(out, camera_label, (w - 20 - 11 * len(camera_label), h - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    bright, bright_ok = light_status(frame)  # last run's photos were too dark and nobody noticed until the end
    cv2.putText(out, f"light {bright:.0f}/255" + ("" if bright_ok else "  DARK: add light"), (w - 330, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0) if bright_ok else (0, 140, 255), 2 if not bright_ok else 1, cv2.LINE_AA)
    cv2.putText(out, "SPACE photo   N new session   Q quit", (16, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    if toast:
        cv2.putText(out, toast, (16, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def run_preview(cam, root: str, session_name: str | None, min_sharpness: float,
                camera_label: str = "",
                show: Callable[[np.ndarray], None] | None = None,
                wait_key: Callable[[], int] | None = None,
                say: Callable[[str], None] = print) -> list[CaptureSession]:
    """The interactive loop. `show` / `wait_key` are injectable so it can be tested
    without a window. Returns every session that was started."""
    show = show or (lambda img: cv2.imshow(WIN, img))
    wait_key = wait_key or (lambda: cv2.waitKey(1) & 0xFF)

    session = CaptureSession(root, session_name)
    sessions = [session]
    say(f"Session folder: {session.dir}")
    say("SPACE = photo, N = new session, Q/ESC = quit")
    toast, toast_until = None, 0.0

    while True:
        frame = cam.read()
        s = sharpness(frame)
        if time.monotonic() > toast_until:
            toast = None
        show(draw_overlay(frame, session.name, session.count, s, min_sharpness, toast, camera_label))

        key = wait_key()
        if key in (ord("q"), ord("Q"), 27):
            break
        if key == ord(" "):
            path = session.save_frame(frame, sharpness=s)
            toast, toast_until = f"saved {path.name}", time.monotonic() + TOAST_SECONDS
            say(f"saved {path}  (sharpness {s:.0f})")
        elif key in (ord("n"), ord("N")):
            if session.count == 0:
                toast, toast_until = "this session is still empty", time.monotonic() + TOAST_SECONDS
            else:
                say(f"{session.count} photo(s) in {session.dir}")
                session = CaptureSession(root)
                sessions.append(session)
                toast, toast_until = "NEW SESSION", time.monotonic() + TOAST_SECONDS
                say(f"New session: {session.dir}")
    for s_ in sessions:
        if s_.count:
            say(f"{s_.count} photo(s) in {s_.dir}")
    return sessions


def cmd_list(snapshot_dir: str = "camera_snapshots") -> int:
    print("Probing camera indices (this briefly switches each camera on)...")
    cams = list_cameras(snapshot_dir=snapshot_dir)
    if not cams:
        print("No cameras found. Is the webcam plugged in, and does your terminal have camera access?\n"
              "(macOS: System Settings > Privacy & Security > Camera.)")
        return 1
    for i, w, h, snap in cams:
        print(f"index {i}: {w}x{h}   snapshot: {snap}")
    print(f"\nOpen the snapshots in {snapshot_dir}/ to see which index is which (cover the built-in "
          "camera or point the C270 at something distinctive), then run:\n"
          "    python -m capture --camera-index N\n"
          "Indices can change when you plug/unplug cameras (or when an iPhone appears as a camera "
          "via Continuity Camera), so re-check if the wrong one opens. To make a choice stick: "
          "export CAMERA_INDEX=N")
    return 0


def cmd_check(args) -> int:
    """Grab ~2 s of video and report on the things that break SfM: wrong
    resolution, low fps, blur, bad exposure. Also saves check_frame.jpg so you
    can eyeball it."""
    with Camera(args.camera, args.width, args.height, lock_exposure=args.lock_exposure) as cam:
        fps = measure_fps(cam, 2.0)
        samples = [frame_stats(cam.read()) for _ in range(15)]
        frame = cam.read_fresh()
        cv2.imwrite("check_frame.jpg", frame)

    st = samples[-1]
    sharp_vals = [s.sharpness for s in samples]
    mean_sharp = sum(sharp_vals) / len(sharp_vals)
    print(f"camera     : index {args.camera}")
    print(f"resolution : {st.width}x{st.height}")
    print(f"frame rate : {fps:.1f} fps")
    print(f"sharpness  : mean {mean_sharp:.0f}  (min {min(sharp_vals):.0f}, max {max(sharp_vals):.0f})")
    print(f"brightness : {st.mean_brightness:.0f}/255   clipped pixels: {st.clipped_pct:.1f}%")
    print("saved      : check_frame.jpg  (open it and confirm the object is crisp)")

    warnings = []
    if (st.width, st.height) != (args.width, args.height):
        warnings.append(f"camera is delivering {st.width}x{st.height}, not {args.width}x{args.height}")
    if fps < 8:
        warnings.append("very low frame rate; try a different USB port/cable (avoid hubs)")
    if st.mean_brightness < MIN_BRIGHTNESS:
        warnings.append(f"image is dark ({st.mean_brightness:.0f}/255, want {MIN_BRIGHTNESS}+); add diffuse light "
                        "from several sides: dark frames are noisy, match poorly and make the model's colours muddy")
    if st.mean_brightness > 200 or st.clipped_pct > 10:
        warnings.append("image is over-exposed / heavily clipped; reduce light or glare")
    if max(sharp_vals) > 0 and min(sharp_vals) < 0.6 * max(sharp_vals):
        warnings.append("sharpness varies a lot between frames; is the scene or camera moving?")
    if mean_sharp < 40:
        warnings.append(
            "low sharpness. Point the camera at the textured object at your real working "
            "distance. The C270 is fixed-focus, so if you are too close it will be soft "
            "regardless of settings; move back until check_frame.jpg is crisp."
        )
    for w in warnings:
        print(f"WARN: {w}")
    print("PASS" if not warnings else "Review the warnings above.")
    return 0 if not warnings else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m capture", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", "--camera-index", "--device", dest="camera", type=int,
                   default=int(os.environ.get("CAMERA_INDEX", 0)),
                   help="camera index to use (default: $CAMERA_INDEX, else 0); see --list-cameras")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--list-cameras", action="store_true")
    p.add_argument("--check", action="store_true", help="run the feed sanity check and exit")
    p.add_argument("--lock-exposure", action="store_true",
                   help="best-effort lock of exposure/white balance (backend dependent)")
    p.add_argument("--min-sharpness", type=float, default=60.0,
                   help="sharpness readout turns red below this Laplacian variance (scene dependent)")
    p.add_argument("--session-root", default="captures")
    p.add_argument("--session-name", default=None,
                   help="default: timestamp like 2026-09-25_2130 (an explicit name resumes that folder)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        if args.list_cameras:
            return cmd_list()
        if args.check:
            return cmd_check(args)
        with Camera(args.camera, args.width, args.height, lock_exposure=args.lock_exposure) as cam:
            try:
                run_preview(cam, args.session_root, args.session_name, args.min_sharpness,
                            camera_label=f"camera {args.camera}  {cam.width}x{cam.height}")
            finally:
                cv2.destroyAllWindows()
        return 0
    except CameraError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
