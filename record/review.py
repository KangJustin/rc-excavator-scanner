"""Stats over local episodes, and interactive review that marks each one good or bad.

The verdict lives in the episode's meta.json under "review" ("good" / "bad"); pi-upload
skips episodes marked bad, since PI has no way to delete an uploaded episode.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .controls import CONTROL_NAMES, ControlState, describe
from .episode import FORMAT_VERSION, Step, load_episode

TARGET_STEPS = 3000        # rough floor from the PI guide ("at least a few thousand steps")
RATE_TOLERANCE = 0.10      # the system/step frequency tolerance in our PI contract (pi.STEP_TOLERANCE)
GAP_FACTOR = 2.0           # a gap this many steps long between frames means a dropped frame
IDLE_WARN_FRACTION = 0.5
MIN_STEPS = 50


@dataclass
class EpisodeStats:
    path: Path
    steps: int
    seconds: float
    fps_target: int
    fps_achieved: float
    idle_fraction: float
    longest_idle_s: float
    review: str | None
    task: str
    max_gap_s: float = 0.0

    @property
    def flags(self) -> list[str]:
        f = []
        if abs(self.fps_achieved - self.fps_target) > RATE_TOLERANCE * self.fps_target:
            f.append("RATE")
        if self.max_gap_s > GAP_FACTOR / self.fps_target:
            f.append("GAP")  # flagged, not excluded: the PI contract's own tolerance decides what is rejected
        if self.idle_fraction > IDLE_WARN_FRACTION:
            f.append("IDLE")
        if self.steps < MIN_STEPS:
            f.append("SHORT")
        return f

    @property
    def usable_steps(self) -> int:
        """Steps that would train: non-idle, from episodes that aren't marked bad or badly timed."""
        if self.review == "bad" or "RATE" in self.flags:
            return 0
        return round(self.steps * (1 - self.idle_fraction))


def episode_stats(path: Path) -> EpisodeStats:
    meta, steps = load_episode(path)
    fps = meta["fps"]
    n = len(steps)
    span = steps[-1].t_ns - steps[0].t_ns
    achieved = (n - 1) * 1e9 / span if n > 1 and span > 0 else float(fps)
    idle = [not np.any(s.action) for s in steps]
    longest = run = 0
    for i in idle:
        run = run + 1 if i else 0
        longest = max(longest, run)
    max_gap = max((b.t_ns - a.t_ns for a, b in zip(steps, steps[1:])), default=0) / 1e9
    return EpisodeStats(path, n, n / fps, fps, achieved, sum(idle) / n, longest / fps,
                        meta.get("review"), meta.get("task", ""), max_gap)


def _metas(root: Path):
    for d in sorted(p for p in Path(root).glob("*") if (p / "meta.json").exists()):
        yield d, json.loads((d / "meta.json").read_text())


def _current_format(m: dict) -> bool:
    return m.get("format_version") == FORMAT_VERSION and m.get("control_names") == list(CONTROL_NAMES)


def find_episodes(root: Path) -> tuple[list[Path], list[Path]]:
    """(usable episode dirs, interrupted-or-empty ones) under root. Usable means finished, has at
    least one step, and matches the current control format; see legacy_episodes for finished
    episodes recorded with a different format_version or control_names (load_episode refuses
    those, so they must never end up in either list here, or a caller that loads every 'usable'
    episode -- like format_stats below -- would crash on one)."""
    good, bad = [], []
    for d, m in _metas(root):
        if m.get("complete") and m.get("steps") and _current_format(m):
            good.append(d)
        elif not m.get("complete") or (not m.get("steps") and _current_format(m)):
            bad.append(d)
    return good, bad


def legacy_episodes(root: Path) -> list[Path]:
    """Finished episodes recorded with a different format_version or control_names. They can't be
    loaded by this code and are never counted as usable."""
    return [d for d, m in _metas(root) if m.get("complete") and not _current_format(m)]


def format_stats(root: Path) -> str:
    complete, incomplete = find_episodes(root)
    rows = [episode_stats(p) for p in complete]
    out = [f"{'episode':<28}{'steps':>6}{'sec':>6}{'Hz':>6}{'idle':>6}  review   flags"]
    for s in rows:
        out.append(f"{s.path.name:<28}{s.steps:>6}{s.seconds:>6.0f}{s.fps_achieved:>6.1f}"
                   f"{s.idle_fraction:>6.0%}  {s.review or '-':<7}  {' '.join(s.flags) or '-'}")
    total = sum(s.steps for s in rows)
    usable = sum(s.usable_steps for s in rows)
    out += ["",
            f"{len(rows)} episodes, {total} steps ({total / max(1, rows[0].fps_target if rows else 1) / 60:.1f} min)"
            if rows else "No complete episodes yet.",
            # review is usually absent (our own recorder) but can be the literal string "unreviewed"
            # (seen from a different recorder), so anything that isn't "good"/"bad" counts as unreviewed.
            f"marked good {sum(s.review == 'good' for s in rows)}, bad {sum(s.review == 'bad' for s in rows)}, "
            f"unreviewed {sum(s.review not in ('good', 'bad') for s in rows)}"]
    if rows:
        out.append(f"usable for training (not bad, rate ok, idle stretches removed): ~{usable} steps "
                   f"of the ~{TARGET_STEPS} minimum ({usable / TARGET_STEPS:.0%})")
        tasks = sorted({s.task for s in rows})
        if len(tasks) > 1:
            out.append(f"NOTE: {len(tasks)} different task strings: {tasks}")
    if incomplete:
        out.append(f"{len(incomplete)} incomplete episode(s) (interrupted, crashed, or saved with 0 steps): "
                   + ", ".join(p.name for p in incomplete))
    legacy = legacy_episodes(root)
    if legacy:
        out.append(f"{len(legacy)} episode(s) in an older or mismatched control format, ignored "
                   "(control_names or format_version does not match the current contract): "
                   + ", ".join(p.name for p in legacy))
    return "\n".join(out)


# ------------------------------------------------------------------ review
def set_review(path: Path, verdict: str | None) -> None:
    if verdict not in ("good", "bad", None):
        raise ValueError("verdict must be 'good', 'bad' or None")
    p = Path(path) / "meta.json"
    meta = json.loads(p.read_text())
    if verdict is None:
        meta.pop("review", None)
    else:
        meta["review"] = verdict
    p.write_text(json.dumps(meta, indent=2))


def is_rejected(path: Path) -> bool:
    return json.loads((Path(path) / "meta.json").read_text()).get("review") == "bad"


def review_overlay(frame: np.ndarray, step: Step, i: int, n: int, fps: int, paused: bool,
                   verdict: str | None) -> np.ndarray:
    img = frame.copy()
    lines = [f"step {i + 1}/{n}  {i / fps:.1f}s {'[PAUSED]' if paused else ''}  verdict: {verdict or '-'}",
             describe(ControlState.from_vector(step.action))]
    for k, text in enumerate(lines):
        cv2.putText(img, text, (10, 25 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, text, (10, 25 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, "SPACE pause  , . step  G good  B bad  N skip  Q quit", (10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return img


def review_episode(path: Path, show: Callable[[np.ndarray], int]) -> str:
    """Play an episode. `show(img)` displays it and returns a key code (-1 for none), waiting
    about one frame period. Returns 'good', 'bad', 'skip' or 'quit'; good/bad are saved."""
    meta, steps = load_episode(path)
    fps, n = meta["fps"], len(steps)
    i, paused, verdict = 0, False, meta.get("review")
    while True:
        frame = cv2.imread(str(steps[i].frame_path))
        if frame is None:
            raise FileNotFoundError(steps[i].frame_path)
        key = show(review_overlay(frame, steps[i], i, n, fps, paused, verdict))
        if key in (ord("g"), ord("b")):
            verdict = "good" if key == ord("g") else "bad"
            set_review(path, verdict)
            return verdict
        if key == ord("n"):
            return "skip"
        if key in (ord("q"), 27):
            return "quit"
        if key == ord(" "):
            paused = not paused
        elif key == ord(",") and paused:
            i = max(0, i - 1)
        elif key == ord(".") and paused:
            i = min(n - 1, i + 1)
        elif not paused:
            if i < n - 1:
                i += 1
            else:
                paused = True  # hold the last frame until you give a verdict
