"""Drive the excavator using a hosted Physical Intelligence checkpoint.

Same loop shape as excavator-collect's collect.py, with the model standing in for a
human's keyboard: read the camera, ask the model what to do next, threshold its answer
at 0.5, and send that to the ESP32.

    python -m record infer                  # Drives, by default. Watch the robot the
                                             # whole time, tracks lifted or a hand on
                                             # the power -- this is an undertrained
                                             # checkpoint (no held-out validation score
                                             # yet) making autonomous decisions with
                                             # nobody confirming each one. Ctrl+C at any
                                             # point releases all four lines.
    python -m record infer --no-drive       # Real camera, real model, but prints what
                                             # it would send instead of sending it.
                                             # Useful the first time you point this at
                                             # a *new* checkpoint, before trusting it.
    python -m record infer --fake           # No camera, no model, no ESP32, no
                                             # PI_API_KEY needed. Exercises the loop
                                             # logic only: chunking, thresholding, the
                                             # max-steps stop, and that pins always
                                             # release even on error.
    python -m record infer --record-video   # Also saves a video of the run to
                                             # captures/infer-<timestamp>.mp4 (or
                                             # --record-video PATH for an explicit one),
                                             # at RECORD_FPS regardless of --chunk-size,
                                             # for capture.best_frames / reconstruct --
                                             # the same CAD-scan pipeline a hand-driven
                                             # orbit_capture.py session feeds. Frames are
                                             # the raw pre-crop camera image, not the
                                             # training-cropped one the model sees.
    python -m record infer --record-frames  # Same idea, but as numbered stills in
                                             # photos/infer-<timestamp>/ (or
                                             # --record-frames DIR), the exact folder
                                             # shape scan2cad.py's --input takes:
                                             #   python scan2cad.py --input photos/infer-<timestamp> \
                                             #     --marker-mm <mm> --quality <q> --describe "<desc>"
                                             # Both --record-video and --record-frames
                                             # can be given together.

--max-steps is still a hard stop (default 100), and Ctrl+C or a normal `kill` (SIGTERM)
both release every line before exiting -- those don't go away with --no-drive removed as
the default; only the extra confirmation flag does. IMPORTANT: the board's actual
firmware (matrix.ino) has NO watchdog of its own (our original excavator_control.ino
design did; matrix.ino does not) -- there is no hardware backstop if this process is
killed in a way that skips its own cleanup. That means: never Ctrl+Z (SIGTSTP) this
process. A suspended process runs no Python code at all, so nothing gets released, and
whatever was last commanded stays in effect indefinitely. If you do hit Ctrl+Z by
accident, run `fg` to bring it back and then Ctrl+C it properly -- don't `kill` a
suspended job; `kill`'s default SIGTERM is now caught same as Ctrl+C, but a `kill -9`
(SIGKILL) cannot be caught by any process and will skip cleanup exactly like SIGTSTP does.

This module deliberately does not duplicate the camera/serial code: it imports
mjpeg.py, pins.py and rawlog.py from the excavator-collect checkout (the one place
that code is authoritative -- a second, drifting copy is exactly the kind of format
mismatch that bit this project once already) via --collect-dir / $EXCAVATOR_COLLECT_DIR.
pi_sdk[inference] lives in *this* project's venv, which is where its registry
credentials already work; excavator-collect's own venv does not have them.
"""
from __future__ import annotations

import argparse
import importlib
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

DEFAULT_COLLECT_DIR = "/Users/justinkang/Downloads/excavator-collect 3"
DEFAULT_GATEWAY_URL = "wss://api.pi-fleet.com/v1/models/ckpt-999-excavator-orbit-v2-test1"
DEFAULT_URL = "http://robot-cam.local:8080/stream"
DEFAULT_PORT = "/dev/cu.usbserial-0001"
DEFAULT_TASK = "orbit around the object"

CAMERA_KEY = "observation/camera/rgb/image"
STATE_KEY = "observation/robot/controls"
ACTION_KEY = "action/robot/controls"

TICK_S = 0.1  # matches the 10 Hz the checkpoint was trained at
STALL_S = 2.0  # give up waiting for a camera frame after this long
RECORD_FPS = round(1 / TICK_S)  # --record-video is written at this rate, one frame per step tick


def _load_hardware_modules(collect_dir: str):
    """Import mjpeg/pins/rawlog from the excavator-collect checkout at runtime, so this
    file has no hard import-time dependency on that folder existing (needed for --fake,
    which runs with neither camera nor ESP32 present)."""
    if collect_dir not in sys.path:
        sys.path.insert(0, collect_dir)
    return (importlib.import_module("mjpeg"), importlib.import_module("pins"),
            importlib.import_module("rawlog"))


def threshold(chunk) -> np.ndarray:
    """Model output (arbitrary floats, e.g. a stray -0.02) -> real on/off decisions.
    Elementwise: >= 0.5 is high, everything else is low."""
    return (np.asarray(chunk, dtype=np.float64) >= 0.5).astype(np.float32)


class Runner:
    """The decide-and-act loop, with every real-world call injected so it can be tested
    with no camera, model, or ESP32.

    `get_frame()` returns one RGB image. `infer_fn(image, state)` returns a raw
    (steps_in_chunk, len(channel_names)) array of model output for the *next several*
    steps (a chunk); only the first `chunk_size` rows of it are used. `send_fn(name,
    level)` is called for each channel only when `drive` is True. `release_fn()` always
    runs when the loop ends, drive or not -- on success, on an exception, and on Ctrl-C
    -- the same belt-and-suspenders release-on-exit every other part of this project
    already follows. `record_fn()`, if given, is called once per step tick (not once per
    model call), so a saved video comes out at the tick rate regardless of --chunk-size.
    """

    def __init__(self, get_frame: Callable[[], np.ndarray], infer_fn, send_fn,
                channel_names: tuple[str, ...], *, chunk_size: int, max_steps: int,
                drive: bool, release_fn: Callable[[], None] | None = None,
                record_fn: Callable[[], None] | None = None,
                sleep: Callable[[float], None] = time.sleep,
                now: Callable[[], float] = time.monotonic, say: Callable[[str], None] = print):
        self.get_frame = get_frame
        self.infer_fn = infer_fn
        self.send_fn = send_fn
        self.channel_names = channel_names
        self.chunk_size = max(1, chunk_size)
        self.max_steps = max_steps
        self.drive = drive
        self.release_fn = release_fn or self._default_release
        self.record_fn = record_fn
        self.sleep = sleep
        self.now = now
        self.say = say
        self.state = np.zeros(len(channel_names), dtype=np.float32)
        self.model_calls = 0

    def _default_release(self) -> None:
        """Matches the loop body's own guard: never touches send_fn unless actually driving."""
        if self.drive:
            for name in self.channel_names:
                self.send_fn(name, False)

    def _describe(self, step: int, action: np.ndarray) -> str:
        levels = "  ".join(f"{n}={'ON' if v else 'off'}" for n, v in zip(self.channel_names, action))
        return f"step {step}: {levels}" + ("" if self.drive else "  (not sent: --drive is not set)")

    def run(self) -> int:
        """Runs until max_steps is reached or an exception propagates. Returns the
        number of steps actually executed. Always releases every line before returning
        or raising."""
        next_tick = self.now()
        steps_run = 0
        try:
            while steps_run < self.max_steps:
                image = self.get_frame()
                chunk = threshold(self.infer_fn(image, self.state))
                self.model_calls += 1
                take = min(self.chunk_size, len(chunk), self.max_steps - steps_run)
                for action in chunk[:take]:
                    if self.drive:
                        for name, level in zip(self.channel_names, action):
                            self.send_fn(name, bool(level))
                    self.say(self._describe(steps_run, action))
                    if self.record_fn is not None:
                        self.record_fn()
                    self.state = action.astype(np.float32)
                    steps_run += 1
                    if steps_run >= self.max_steps:
                        break
                    next_tick += TICK_S
                    delay = next_tick - self.now()
                    if delay > 0:
                        self.sleep(delay)
                    else:
                        next_tick = self.now()  # fell behind; don't try to catch up in a burst
        finally:
            self.release_fn()
            self.say("released all lines")
        return steps_run


# ---------------------------------------------------------------------------- fake mode
def run_fake(max_steps: int, chunk_size: int, channel_names: tuple[str, ...]) -> dict:
    """No camera, no model, no ESP32, no PI_API_KEY. A scripted model alternates between
    an all-high and an all-low chunk, so the test can check thresholding, chunking, and
    that every line is released at the end -- entirely offline."""
    sent: list[tuple[str, bool]] = []
    calls = {"n": 0}

    def get_frame():
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def infer_fn(image, state):
        calls["n"] += 1
        value = 1.0 if calls["n"] % 2 else 0.0
        return np.full((chunk_size, len(channel_names)), value)

    def send_fn(ch, level):
        sent.append((ch, bool(level)))

    runner = Runner(get_frame, infer_fn, send_fn, channel_names, chunk_size=chunk_size,
                    max_steps=max_steps, drive=True, sleep=lambda s: None)
    steps = runner.run()
    return {"steps": steps, "model_calls": calls["n"], "sent": sent}


# ---------------------------------------------------------------------------- real mode
def make_infer_fn(gateway_url: str, task: str, chunk_size: int):
    """Connects to the hosted checkpoint now (so a bad gateway_url or missing
    PI_API_KEY fails immediately, not on the first camera frame) and returns
    (infer_fn, close_fn)."""
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())  # else wss:// fails to verify
    from pi_sdk.inference import ClientConfig, InferenceInput, PolicyClient

    if "PI_API_KEY" not in os.environ:
        raise SystemExit("PI_API_KEY is not set. `source .pi/api_key.env` first.")
    config = ClientConfig(gateway_url=gateway_url, api_key=os.environ["PI_API_KEY"],
                          robot_task_string=task, action_keys=[ACTION_KEY],
                          execute_chunk_size=chunk_size)
    policy = PolicyClient(config)
    policy.connect()

    def infer_fn(image: np.ndarray, state: np.ndarray) -> np.ndarray:
        obs = InferenceInput(images={CAMERA_KEY: image}, states={STATE_KEY: state}, raw_text=task)
        return policy.infer(obs).actions_by_key[ACTION_KEY]

    return infer_fn, policy.close


SHOW_WINDOW = "record infer -- camera"


def make_camera_fns(collect_dir: str, url: str, show: bool = False, record_path: Path | None = None,
                    frames_dir: Path | None = None):
    """Real camera reader + a frame prep matching collect.py's own training_frame crop,
    so the model sees exactly the shape it was trained on.

    `show`, if set, opens a live window on the RAW (pre-crop) frame -- the same one
    connection already being read for inference, not a second stream connection. The
    Pi's camera only allows one viewer at a time (confirmed the hard way more than
    once), so this is the only way to watch the feed while this script is also running,
    short of a second machine with its own connection.

    `record_path` and/or `frames_dir`, if given, write every frame `record_tick()` is
    called on -- the raw pre-crop image, for more detail than the model's
    training-cropped view -- over that same one connection: `record_path` to an .mp4 at
    RECORD_FPS, `frames_dir` as numbered stills (0001.jpg, ...), scan2cad.py's own
    `--input photos/<title>` folder shape, so its select_frames.py keyframe picker runs
    instead of the weaker one scan2cad falls back to for a bare video file. Returns
    (get_frame, record_tick, stop)."""
    mjpeg, _pins, rawlog = _load_hardware_modules(collect_dir)
    import cv2

    reader = mjpeg.LatestFrame(url)
    reader.start()
    if show:
        cv2.namedWindow(SHOW_WINDOW, cv2.WINDOW_NORMAL)
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    dropped = {"n": 0}
    frame_count = {"n": 0}

    def _grab_bgr() -> np.ndarray | None:
        payload, _t_ns, _error = reader.latest()
        if not payload:
            return None
        array = np.frombuffer(payload, dtype=np.uint8).copy()
        return cv2.imdecode(array, cv2.IMREAD_COLOR)

    def _write_video_frame(bgr: np.ndarray) -> None:
        nonlocal writer
        h, w = bgr.shape[:2]
        even = bgr[: h - h % 2, : w - w % 2]  # matches record/pi.py's write_shards: H.264 needs even sizes
        if writer is None:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(record_path), cv2.VideoWriter_fourcc(*"mp4v"), RECORD_FPS,
                                     (even.shape[1], even.shape[0]))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open {record_path} for writing")
        writer.write(even)

    def _write_frame_jpg(bgr: np.ndarray) -> None:
        frame_count["n"] += 1
        path = frames_dir / f"{frame_count['n']:04d}.jpg"
        if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Failed to write {path}")

    def get_frame() -> np.ndarray:
        deadline = time.monotonic() + STALL_S
        error = None
        while time.monotonic() < deadline:
            payload, t_ns, error = reader.latest()
            if payload:
                array = np.frombuffer(payload, dtype=np.uint8).copy()
                bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError("jpeg decode failed")
                if show:
                    cv2.imshow(SHOW_WINDOW, bgr)
                    cv2.waitKey(1)  # pumps the window's event loop; doesn't block
                h, w = bgr.shape[:2]
                if (w, h) != rawlog.FRAME_SIZE:
                    if (w, h) != (1280, 720):
                        raise RuntimeError(f"unexpected camera frame size {w}x{h}")
                    bgr = cv2.resize(bgr[:, 160:1120], rawlog.FRAME_SIZE, interpolation=cv2.INTER_AREA)
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            time.sleep(0.05)
        raise RuntimeError(f"camera stalled: {error}")

    def record_tick() -> None:
        """Best-effort: a momentary gap in the mjpeg cache is dropped and counted, not
        fatal -- recording the scan is secondary to driving the robot."""
        if record_path is None and frames_dir is None:
            return
        bgr = _grab_bgr()
        if bgr is None:
            dropped["n"] += 1
            return
        if record_path is not None:
            _write_video_frame(bgr)
        if frames_dir is not None:
            _write_frame_jpg(bgr)

    def stop():
        reader.stop()
        if show:
            cv2.destroyWindow(SHOW_WINDOW)
        if writer is not None:
            writer.release()

    record_tick.dropped = dropped
    record_tick.frame_count = frame_count
    return get_frame, record_tick, stop


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m record infer", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=DEFAULT_URL, help="Pi camera MJPEG stream")
    p.add_argument("--port", default=DEFAULT_PORT, help="ESP32 serial port")
    p.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--chunk-size", type=int, default=5,
                   help="predicted actions to run per model call (default 5). Bigger amortizes "
                        "the ~0.5s call latency better but acts on staler visual context for "
                        "the later steps in the chunk.")
    p.add_argument("--max-steps", type=int, default=100,
                   help="hard stop, so a first run can't run away (default 100, ~10-20s "
                        "depending on chunk size and real latency)")
    p.add_argument("--no-drive", action="store_true",
                   help="observe only: the real camera and real model still run, but nothing "
                        "is sent to the ESP32. Driving is the default; pass this to hold off, "
                        "e.g. the first time you point --gateway-url at a new checkpoint.")
    p.add_argument("--collect-dir", default=os.environ.get("EXCAVATOR_COLLECT_DIR", DEFAULT_COLLECT_DIR),
                   help="path to the excavator-collect checkout (for mjpeg.py/pins.py/rawlog.py)")
    p.add_argument("--show", action="store_true",
                   help="open a live window on the raw camera feed while running -- the same "
                        "connection already in use, not a second one (the Pi only allows one "
                        "viewer at a time). Use this to actually see what the model sees.")
    p.add_argument("--fake", action="store_true",
                   help="no camera, model, or ESP32: test the loop logic only")
    p.add_argument("--record-video", nargs="?", const="auto", default=None, metavar="PATH",
                   help="also save the run to an .mp4 (raw pre-crop frames, one per step tick, "
                        "independent of --chunk-size) for capture.best_frames / reconstruct -- "
                        "the same CAD-scan pipeline orbit_capture.py feeds. Give a path, or pass "
                        "with no path to auto-name it captures/infer-<timestamp>.mp4")
    p.add_argument("--record-frames", nargs="?", const="auto", default=None, metavar="DIR",
                   help="also save the run as numbered stills (0001.jpg, ...; one per step tick), "
                        "in the exact photos/<title> folder shape scan2cad.py's --input takes -- "
                        "that's what makes it run scan2cad's select_frames.py keyframe picker "
                        "instead of its weaker built-in one for a bare video. Give a folder, or "
                        "pass with no path to auto-name it photos/infer-<timestamp>/")
    return p.parse_args(argv)


def _raise_keyboard_interrupt(signum, frame):
    """SIGTERM's default action is to terminate the process immediately, running no
    Python cleanup at all -- unlike SIGINT (Ctrl+C), which Python's own default handler
    already turns into a catchable KeyboardInterrupt. Installing this for SIGTERM makes
    a plain `kill` behave the same safe way Ctrl+C already does: it propagates through
    Runner.run()'s try/finally (and main()'s own), releasing every line before exit.
    This cannot help with SIGTSTP (Ctrl+Z) or SIGKILL (`kill -9`) -- neither one runs any
    Python code at all, caught or not."""
    raise KeyboardInterrupt("received SIGTERM")


def main(argv=None) -> int:
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    args = parse_args(argv)
    drive = not args.no_drive

    if args.fake:
        _mjpeg, _pins, rawlog = _load_hardware_modules(args.collect_dir)
        result = run_fake(args.max_steps, args.chunk_size, rawlog.CHANNELS)
        print(f"ran {result['steps']} fake steps, {result['model_calls']} model calls, "
             f"{len(result['sent'])} pin writes, last 4 sent: {result['sent'][-4:]}")
        return 0

    if drive:
        print("*** This WILL send commands to the ESP32 and move the excavator. ***")
        print("*** Lift the tracks off the ground, or keep a hand on the power, before "
             "continuing. Ctrl+C releases all four lines at any point. ***")
    else:
        print("*** --no-drive: observing only, nothing will be sent to the ESP32. ***")

    record_path = None
    if args.record_video is not None:
        record_path = (Path("captures") / f"infer-{datetime.now():%Y-%m-%d_%H%M%S}.mp4"
                       if args.record_video == "auto" else Path(args.record_video))
        print(f"*** Recording to {record_path} for capture.best_frames / reconstruct. ***")

    frames_dir = None
    if args.record_frames is not None:
        frames_dir = (Path("photos") / f"infer-{datetime.now():%Y-%m-%d_%H%M%S}"
                     if args.record_frames == "auto" else Path(args.record_frames))
        print(f"*** Saving stills to {frames_dir} for `python scan2cad.py --input {frames_dir} ...`. ***")

    _mjpeg, pins_mod, rawlog = _load_hardware_modules(args.collect_dir)
    get_frame, record_tick, stop_camera = make_camera_fns(
        args.collect_dir, args.url, show=args.show, record_path=record_path, frames_dir=frames_dir)
    infer_fn, close_policy = make_infer_fn(args.gateway_url, args.task, args.chunk_size)
    pins = pins_mod.Pins(args.port) if drive else None

    def send_fn(ch, level):
        pins.set_channel(ch, level)

    def release_fn():
        if pins is not None:
            pins.release()

    recording = record_path is not None or frames_dir is not None
    runner = Runner(get_frame, infer_fn, send_fn, rawlog.CHANNELS, chunk_size=args.chunk_size,
                    max_steps=args.max_steps, drive=drive, release_fn=release_fn,
                    record_fn=record_tick if recording else None)
    try:
        steps = runner.run()
        print(f"ran {steps} steps ({runner.model_calls} model calls)")
        if record_path is not None:
            print(f"saved video to {record_path} ({record_tick.dropped['n']} dropped frames)")
            print(f"Next: python -m capture.best_frames {record_path}")
        if frames_dir is not None:
            print(f"saved {record_tick.frame_count['n']} stills to {frames_dir} "
                 f"({record_tick.dropped['n']} dropped frames)")
            print(f"Next: python scan2cad.py --input {frames_dir} --marker-mm <marker size in mm> "
                 f"--quality <low|medium|high|max> --describe \"<what the object is>\"")
        return 0
    finally:
        stop_camera()
        close_policy()
        if pins is not None:
            pins.close()


if __name__ == "__main__":
    sys.exit(main())
