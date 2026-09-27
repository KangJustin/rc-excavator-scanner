"""The raw control log, and the rule that turns it into one action per step.

controls_raw.jsonl has one line per edge, on the same clock as steps.jsonl's t_ns:

    {"seq": 0, "t_ns": 1790440304812345000, "ch": "A", "level": 0, "init": true}   # starting level
    {"seq": 2, "t_ns": 1790440304912000000, "ch": "A", "level": 1}                 # a change

`seq` orders lines that share a timestamp (more than one can change on the same tick). The `init`
lines give each line's level at the start, so the log can be replayed without anything else.

PI takes exactly one array per topic per step, so this log stays local. It is lossless: a tap
shorter than half a step is 0 in the per-step vector but is still here, and the step rate could
be changed later without re-recording.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .controls import CONTROL_NAMES, ControlState

HIGH_FRACTION = 0.5  # a line counts as high for a step if it was high for at least this share of it


class RawLog:
    """Append-only writer. Call start() once, then update() with the commanded levels each tick;
    only the lines that changed are written."""

    def __init__(self, path: Path):
        self._f = open(path, "w")
        self._levels: tuple[bool, ...] | None = None
        self._seq = 0

    def start(self, t_ns: int, state: ControlState) -> None:
        for name, on in zip(CONTROL_NAMES, state.levels):
            self._write(t_ns, name, on, init=True)
        self._levels = state.levels

    def update(self, t_ns: int, state: ControlState) -> None:
        if self._levels is None:
            raise RuntimeError("RawLog.start() must be called first")
        for name, was, on in zip(CONTROL_NAMES, self._levels, state.levels):
            if on != was:
                self._write(t_ns, name, on)
        self._levels = state.levels

    def _write(self, t_ns: int, name: str, on: bool, init: bool = False) -> None:
        line = {"seq": self._seq, "t_ns": int(t_ns), "ch": name, "level": int(on)}
        if init:
            line["init"] = True
        self._f.write(json.dumps(line) + "\n")
        self._f.flush()
        self._seq += 1

    def close(self) -> None:
        self._f.close()


def load_edges(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _high_fraction(edges: list[tuple[int, int]], a: int, b: int) -> float:
    """Share of [a, b) that a line spent high. `edges` is [(t_ns, level)] in log order."""
    level = 0
    for t, lv in edges:  # level already in effect at a
        if t > a:
            break
        level = lv
    high, cursor = 0, a
    for t, lv in edges:
        if t <= a:
            continue
        if t >= b:
            break
        if level:
            high += t - cursor
        cursor, level = t, lv
    if level:
        high += b - cursor
    return high / (b - a)


def derive_actions(edges: list[dict], step_t_ns: list[int], period_ns: int) -> np.ndarray:
    """Per-step action from the raw log: shape (steps, lines), float32, 0.0 or 1.0.

    Step i covers [t_i, t_{i+1}); the last step covers one step period. A line is high for the
    step if it was high for at least HIGH_FRACTION of that window."""
    per_line = {n: [(e["t_ns"], e["level"]) for e in edges if e["ch"] == n] for n in CONTROL_NAMES}
    ends = list(step_t_ns[1:]) + [step_t_ns[-1] + period_ns] if step_t_ns else []
    out = np.zeros((len(step_t_ns), len(CONTROL_NAMES)), dtype=np.float32)
    for i, (a, b) in enumerate(zip(step_t_ns, ends)):
        for j, name in enumerate(CONTROL_NAMES):
            out[i, j] = 1.0 if _high_fraction(per_line[name], a, b) >= HIGH_FRACTION else 0.0
    return out
