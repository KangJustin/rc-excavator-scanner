"""python -m record <command>

    run          teleoperate the excavator through the ESP32 and record an episode
    stats        summarize recorded episodes (steps, Hz, idle fraction, how far to a dataset)
    review       play episodes back and mark each good or bad (bad ones are never uploaded)
    pi-setup     create the PI platform contract + training profile (once)
    pi-upload    convert recorded episodes to MCAP and upload them to PI
    pi-finetune  start (and optionally wait for) a fine-tune on the uploaded episodes
    infer        run a hosted checkpoint against the camera; --drive to actually move it

    python -m record run --dry-run --fake-camera        # no hardware at all
    python -m record run --port /dev/cu.usbserial-0001 --camera 1 --task "orbit the coke can"
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2

from capture import Camera, CameraError, FakeCamera
from control import ControlError, ExcavatorConfig, make_excavator

from .controls import HELP, KEY_DISCARD, KEY_SAVE
from .episode import DEFAULT_FPS, overlay, run_episode


def _cmd_run(args) -> int:
    cfg = ExcavatorConfig.from_file(args.config) if args.config else None
    camera = FakeCamera() if args.fake_camera else Camera(args.camera, args.width, args.height,
                                                          lock_exposure=args.lock_exposure)
    root = Path(args.episodes_root)
    root.mkdir(parents=True, exist_ok=True)
    print("Each episode: " + HELP + ". Ctrl-C stops everything.")
    try:
        with camera, make_excavator(args.port, args.dry_run, cfg) as exc:
            while True:
                episode_id = args.episode_id or datetime.now().strftime(f"{args.rig_id}-%Y%m%d-%H%M%S")
                fps = args.fps

                def get_key(frame, state, n):
                    cv2.imshow("record", overlay(frame, state, n, fps))
                    return cv2.waitKey(1) & 0xFF if cv2.getWindowProperty("record", cv2.WND_PROP_VISIBLE) >= 0 else KEY_DISCARD

                summary = run_episode(exc, camera, root / episode_id, task=args.task, fps=fps,
                                      get_key=get_key, max_steps=args.max_steps)
                if summary is None:
                    print("Discarded.")
                else:
                    print(f"Saved {summary.path}: {summary.steps} steps at {summary.achieved_fps:.1f} Hz "
                          f"(target {fps}).")
                    if abs(summary.achieved_fps - fps) > 0.1 * fps:
                        print("  WARNING: more than 10% off the target rate; PI will reject this episode. "
                              "Lower --fps or --camera load.", file=sys.stderr)
                    if summary.max_gap_s > 2 / fps:
                        print(f"  WARNING: longest gap between frames was {summary.max_gap_s * 1000:.0f} ms "
                              f"(a step is {1000 / fps:.0f} ms): a dropped frame. Review it before uploading.",
                              file=sys.stderr)
                if args.once or args.episode_id or input("Record another? [Y/n] ").strip().lower() == "n":
                    return 0
    except (ControlError, CameraError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    finally:
        cv2.destroyAllWindows()


def _cmd_setup(args) -> int:
    from . import pi
    ids = pi.setup(args.fps)
    print("Contract + profile ready:", ids)
    return 0


def _cmd_upload(args) -> int:
    from . import pi
    from .review import is_rejected
    for ep in args.episodes:
        if not (Path(ep) / "meta.json").exists():
            continue  # a glob can match stray files
        if is_rejected(Path(ep)):
            print(f"skipped {Path(ep).name}: marked bad in review")
            continue
        try:
            print("uploaded", pi.upload_episode(Path(ep), robot_id=args.robot_id, success=not args.failed))
        except ValueError as e:  # old-format or interrupted episode
            print(f"skipped {Path(ep).name}: {e}", file=sys.stderr)
    return 0


def _cmd_stats(args) -> int:
    from .review import format_stats
    print(format_stats(Path(args.episodes_root)))
    return 0


def _cmd_review(args) -> int:
    from .review import find_episodes, review_episode
    paths = [Path(p) for p in args.episodes] or find_episodes(Path(args.episodes_root))[0]
    if args.unreviewed:
        import json
        paths = [p for p in paths if "review" not in json.loads((p / "meta.json").read_text())]
    if not paths:
        print("Nothing to review.")
        return 0
    period_ms = max(1, int(1000 / args.fps))
    try:
        for p in paths:
            print(f"reviewing {p.name} ...", end=" ", flush=True)
            result = review_episode(p, lambda img: (cv2.imshow("review", img), cv2.waitKey(period_ms) & 0xFF)[1])
            print(result)
            if result == "quit":
                break
    finally:
        cv2.destroyAllWindows()
    return 0


def _cmd_finetune(args) -> int:
    from . import pi
    run_id = pi.start_finetune(args.name, num_steps=args.num_steps, base_model=args.base_model,
                               batch_size=args.batch_size, episode_ids=args.episode_ids)
    print("started fine-tune", run_id)
    if args.wait:
        pi.wait_for_finetune(run_id)
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["infer"]:
        from . import infer
        return infer.main(argv[1:])

    p = argparse.ArgumentParser(prog="python -m record", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="teleoperate + record episodes")
    r.add_argument("--task", default="orbit around the object")
    r.add_argument("--fps", type=int, default=DEFAULT_FPS)
    r.add_argument("--camera", type=int, default=0)
    r.add_argument("--width", type=int, default=640, help="requested frame width (camera may deliver another size)")
    r.add_argument("--height", type=int, default=480, help="requested frame height")
    r.add_argument("--lock-exposure", action="store_true",
                   help="try to fix exposure/white balance so brightness doesn't drift during an orbit")
    r.add_argument("--fake-camera", action="store_true")
    r.add_argument("--port", default=os.environ.get("EXCAVATOR_PORT"))
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--config")
    r.add_argument("--episodes-root", default="episodes")
    r.add_argument("--rig-id", default=os.environ.get("RIG_ID", "excavator"),
                   help="prefix of generated episode ids (default: $RIG_ID, else 'excavator'); give each "
                        "person or rig its own so ids never collide in PI")
    r.add_argument("--episode-id")
    r.add_argument("--max-steps", type=int)
    r.add_argument("--once", action="store_true", help="record a single episode then exit")
    r.set_defaults(fn=_cmd_run)

    st = sub.add_parser("stats", help="summarize recorded episodes")
    st.add_argument("--episodes-root", default="episodes")
    st.set_defaults(fn=_cmd_stats)

    rv = sub.add_parser("review", help="play episodes back and mark each good or bad")
    rv.add_argument("episodes", nargs="*", help="episode dirs (default: all in --episodes-root)")
    rv.add_argument("--episodes-root", default="episodes")
    rv.add_argument("--unreviewed", action="store_true", help="only episodes with no verdict yet")
    rv.add_argument("--fps", type=int, default=DEFAULT_FPS, help="playback rate")
    rv.set_defaults(fn=_cmd_review)

    s = sub.add_parser("pi-setup", help="create PI platform contract + training profile")
    s.add_argument("--fps", type=int, default=DEFAULT_FPS)
    s.set_defaults(fn=_cmd_setup)

    u = sub.add_parser("pi-upload", help="upload recorded episodes to PI")
    u.add_argument("episodes", nargs="+")
    u.add_argument("--robot-id", default="rc-excavator-1")
    u.add_argument("--failed", action="store_true", help="mark these episodes as unsuccessful")
    u.set_defaults(fn=_cmd_upload)

    f = sub.add_parser("pi-finetune", help="start a PI fine-tune")
    f.add_argument("name", help="lowercase-kebab-case run name")
    f.add_argument("--num-steps", type=int)
    f.add_argument("--batch-size", type=int)
    f.add_argument("--base-model", default="pi07_cosmic")
    f.add_argument("--episode-ids", nargs="+", metavar="ID",
                   help="train on exactly these episodes instead of every successful one "
                        "on the contract (creates a dedicated dataset filter for this run)")
    f.add_argument("--wait", action="store_true")
    f.set_defaults(fn=_cmd_finetune)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
