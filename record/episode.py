"""Record one teleoperated episode to disk (raw, PI-independent), and read it back.

Layout of episodes/<episode_id>/:

    meta.json          episode id, task, fps, format_version, control names, complete flag, step count
    steps.jsonl        one line per step: {"i", "t_ns", "state": [N], "action": [N]}
    controls_raw.jsonl every line's edges, timestamped (see rawlog.py); stays local
    frames/000000.jpg ...   the camera frame at each step (BGR JPEG, quality 95)

Step k means: frame k was grabbed, the state is what had been commanded before it
(zeros at k=0), and the action is what the operator commanded at this tick.

format_version 2 = the vector of digital lines named in record/controls.py (CONTROL_NAMES; W, O, L,
A today). load_episode checks both format_version and control_names, so a mismatch (an older
10-value stick-and-button recording, or a different line set) is refused with a clear error
instead of silently loading the wrong shape.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from control.api import Excavator

from .controls import (
    CONTROL_NAMES, HELP, KEY_DISCARD, KEY_SAVE, ControlState, apply_key, describe,
)
from .rawlog import RawLog

log = logging.getLogger(__name__)

DEFAULT_FPS = 10  # PI needs system/step at a whole number of Hz; 10 is comfortable for a C270
FORMAT_VERSION = 2


def apply_to_robot(exc: Excavator, prev: ControlState, new: ControlState) -> None:
    """Send only what changed, then keepalive. Lines stay where they are set, so a stalled loop
    stops sending keepalives and the firmware watchdog drops all lines."""
    for name in CONTROL_NAMES:
        if new.level(name) != prev.level(name):
            exc.set_line(name, new.level(name))
    exc.keepalive()


@dataclass
class EpisodeSummary:
    path: Path
    steps: int
    achieved_fps: float
    max_gap_s: float = 0.0


def run_episode(exc: Excavator, camera, out_dir: Path, *, task: str, fps: int = DEFAULT_FPS,
                get_key: Callable[[np.ndarray, ControlState, int], int],
                max_steps: int | None = None,
                clock_ns: Callable[[], int] = time.time_ns,
                sleep: Callable[[float], None] = time.sleep) -> EpisodeSummary | None:
    """Run the control+record loop until KEY_SAVE (returns summary) or KEY_DISCARD
    (deletes the episode and returns None). `get_key(frame, state, step)` shows the frame
    however it likes and returns a key code or -1. Robot goes back to neutral on exit."""
    period_ns = int(1e9 / fps)
    out_dir = Path(out_dir)
    (out_dir / "frames").mkdir(parents=True, exist_ok=False)
    meta = {"episode_id": out_dir.name, "task": task, "fps": fps, "format_version": FORMAT_VERSION,
            "control_names": list(CONTROL_NAMES), "complete": False, "steps": 0}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    state = ControlState()      # what was last commanded == the observed state
    raw = RawLog(out_dir / "controls_raw.jsonl")
    n = 0
    first_ns = last_ns = 0
    max_gap_ns = 0
    saved = False
    try:
        with open(out_dir / "steps.jsonl", "w") as steps_f:
            next_tick = clock_ns()
            while max_steps is None or n < max_steps:
                frame = camera.read_fresh()
                t_ns = clock_ns()
                if n == 0:
                    meta["frame_size"] = [frame.shape[1], frame.shape[0]]
                    raw.start(t_ns, state)
                elif list(frame.shape[1::-1]) != meta["frame_size"]:
                    raise RuntimeError(f"Camera frame size changed mid-episode "
                                       f"({meta['frame_size']} -> {list(frame.shape[1::-1])}); PI needs it constant.")
                else:
                    max_gap_ns = max(max_gap_ns, t_ns - last_ns)
                key = get_key(frame, state, n)
                if key == KEY_SAVE:
                    saved = True
                    break
                if key == KEY_DISCARD:
                    break
                action = apply_key(state, key)
                raw.update(clock_ns(), action)  # the edge is stamped when it is commanded, not rounded to a tick
                apply_to_robot(exc, state, action)
                cv2.imwrite(str(out_dir / "frames" / f"{n:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                steps_f.write(json.dumps({"i": n, "t_ns": t_ns, "state": state.to_vector().tolist(),
                                          "action": action.to_vector().tolist()}) + "\n")
                steps_f.flush()
                if n == 0:
                    first_ns = t_ns
                last_ns = t_ns
                state = action
                n += 1
                next_tick += period_ns
                delay = (next_tick - clock_ns()) / 1e9
                if delay > 0:
                    sleep(delay)
                else:
                    next_tick = clock_ns()  # fell behind; don't try to catch up in a burst
    finally:
        raw.close()
        try:
            exc.stop()
        except Exception:
            log.exception("Failed to send NEUTRAL after recording!")

    if max_steps is not None and n >= max_steps:
        saved = True
    if not saved or n == 0:
        shutil.rmtree(out_dir, ignore_errors=True)
        return None
    achieved = (n - 1) * 1e9 / (last_ns - first_ns) if n > 1 and last_ns > first_ns else float(fps)
    meta.update(complete=True, steps=n, achieved_fps=round(achieved, 3), max_gap_ms=round(max_gap_ns / 1e6, 1))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return EpisodeSummary(out_dir, n, achieved, max_gap_ns / 1e9)


@dataclass
class Step:
    t_ns: int
    state: np.ndarray
    action: np.ndarray
    frame_path: Path


def load_episode(path: Path) -> tuple[dict, list[Step]]:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    if not meta.get("complete"):
        raise ValueError(f"{path} is not a complete episode (recording was interrupted).")
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path} was recorded in an older format (format_version {meta.get('format_version')!r}, "
                         f"we need {FORMAT_VERSION}); it does not fit the current contract. Move it out of the "
                         "episodes folder or re-record it.")
    if meta.get("control_names") != list(CONTROL_NAMES):
        raise ValueError(f"{path} was recorded with control_names {meta.get('control_names')!r}, but this code "
                         f"expects {list(CONTROL_NAMES)!r}. Recorder and analysis code are out of sync: update "
                         "one side to match, or re-record.")
    steps = []
    for line in (path / "steps.jsonl").read_text().splitlines():
        d = json.loads(line)
        steps.append(Step(d["t_ns"], np.array(d["state"], dtype=np.float32),
                          np.array(d["action"], dtype=np.float32), path / "frames" / f"{d['i']:06d}.jpg"))
    if len(steps) != meta["steps"]:
        raise ValueError(f"{path}: meta says {meta['steps']} steps but steps.jsonl has {len(steps)}.")
    if not steps:
        raise ValueError(f"{path} has 0 steps (likely an interrupted or crashed recording, saved anyway). "
                         "There is nothing here to review or upload; delete it or re-record.")
    return meta, steps


def overlay(frame: np.ndarray, state: ControlState, n: int, fps: int) -> np.ndarray:
    img = frame.copy()
    lines = [f"REC step {n}  ({n / fps:.0f}s)", describe(state)]
    for i, text in enumerate(lines):
        cv2.putText(img, text, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, text, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, HELP, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return img
