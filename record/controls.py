"""The excavator's control vector and the keyboard teleop that produces it.

One control vector = 4 independent digital lines, as float32 in a fixed order:

    W   0.0 low / 1.0 high   tracks forward (drive)
    O   0.0 low / 1.0 high   arm up
    L   0.0 low / 1.0 high   camera tilt
    A   0.0 low / 1.0 high   rotate tracks (steer; used together with W to turn while driving)

There is no reverse, arm-down, or other-direction tilt/turn control yet: each line is a single
on/off switch, not a stick. The lines are independent, e.g. W and A together drive in an arcing
turn, and O or L can be held regardless of what the tracks are doing.

The same vector is logged as the *action* (what the operator commanded at this tick) and,
one tick later, as the *state* (what was last commanded). There is no feedback from the machine.

Keyboard teleop is latched: pressing a line's key toggles that line, and it stays there until
you toggle it back or press SPACE. OpenCV's waitKey has no key-up events, so "hold to drive"
isn't possible.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from control.config import LINE_NAMES

CONTROL_NAMES = LINE_NAMES
N_CONTROLS = len(CONTROL_NAMES)

# What each line does, for the overlays and --help. Keep in sync with control/config.py.
LINE_LABELS = {"W": "drive", "O": "arm up", "L": "cam tilt", "A": "steer"}

# key -> line toggled. Each line's own lowercase letter, so the mapping is self-explanatory.
LINE_KEYS = {ord(name.lower()): name for name in CONTROL_NAMES}
KEY_NEUTRAL = ord(" ")
KEY_SAVE = ord("x")
KEY_DISCARD = 27  # ESC

HELP = ("  ".join(f"{n.lower()} toggle {n} ({LINE_LABELS[n]})" for n in CONTROL_NAMES)
        + "  SPACE all low  X save+end  ESC discard")


@dataclass(frozen=True)
class ControlState:
    levels: tuple[bool, ...] = (False,) * N_CONTROLS

    def to_vector(self) -> np.ndarray:
        return np.array([1.0 if on else 0.0 for on in self.levels], dtype=np.float32)

    @classmethod
    def from_vector(cls, v) -> "ControlState":
        v = [float(x) for x in v]
        if len(v) != N_CONTROLS:
            raise ValueError(f"control vector needs {N_CONTROLS} values, got {len(v)}")
        return cls(tuple(x >= 0.5 for x in v))

    def level(self, name: str) -> bool:
        return self.levels[CONTROL_NAMES.index(name)]


def describe(state: ControlState) -> str:
    """One line for the overlays: 'W HIGH  O low  L low  A HIGH'."""
    return "  ".join(f"{n} {'HIGH' if state.level(n) else 'low'}" for n in CONTROL_NAMES)


def apply_key(state: ControlState, key: int) -> ControlState:
    """Return the new control state after a keypress (unknown keys change nothing)."""
    if key == KEY_NEUTRAL:
        return ControlState()
    if key in LINE_KEYS:
        i = CONTROL_NAMES.index(LINE_KEYS[key])
        levels = list(state.levels)
        levels[i] = not levels[i]
        return replace(state, levels=tuple(levels))
    return state
