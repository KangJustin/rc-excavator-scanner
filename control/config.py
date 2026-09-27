"""Mapping from motions to transmitter controls, plus calibration numbers.

The excavator has no odometry, so every distance/angle is converted to a *duration*
using the calibration constants below. The defaults are placeholders: measure your
own (README > Calibration) and put them in a JSON file passed with --config.
Get the current defaults as a starting point with `python -m control print-config`.

Sign conventions (so `invert` means the same thing everywhere):
    throttle axis  + = forward
    steer axis     + = turn left  (counter-clockwise seen from above)
    turret axis    + = rotate counter-clockwise seen from above
    arm axis       + = increase arm `position` (0.0 .. 1.0; you choose which physical end is 0)
`invert` flips which way the *stick* (or which button) is used to get that motion.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Union

STICK_NAMES = ("left_x", "left_y", "right_x", "right_y")
STICK_MIN, STICK_NEUTRAL, STICK_MAX = 0, 128, 255

# The digital lines the recorder drives directly, one per physical control:
#   W   tracks forward (drive)
#   O   arm up
#   L   camera tilt
#   A   rotate tracks (steer, used together with W to turn while driving)
# Each is high or low; there is no reverse, arm-down, or other-direction tilt/turn control yet.
LINE_NAMES = ("W", "O", "L", "A")


@dataclass(frozen=True)
class StickAxis:
    stick: str
    invert: bool = False

    def __post_init__(self):
        if self.stick not in STICK_NAMES:
            raise ValueError(f"unknown stick {self.stick!r}; expected one of {STICK_NAMES}")


@dataclass(frozen=True)
class ButtonAxis:
    """A motion done with a pair of pushbuttons (plus / minus)."""
    plus: str
    minus: str


Axis = Union[StickAxis, ButtonAxis]


def axis_from_dict(d: dict) -> Axis:
    kind = d.get("type")
    if kind == "stick":
        return StickAxis(d["stick"], d.get("invert", False))
    if kind == "buttons":
        return ButtonAxis(d["plus"], d["minus"])
    raise ValueError(f"axis needs type 'stick' or 'buttons', got {d!r}")


def axis_to_dict(a: Axis) -> dict:
    if isinstance(a, StickAxis):
        return {"type": "stick", "stick": a.stick, "invert": a.invert}
    return {"type": "buttons", "plus": a.plus, "minus": a.minus}


@dataclass
class ExcavatorConfig:
    # --- which physical control does what.  TODO(calibrate): these are GUESSES;
    # --- verify with `python -m control set-stick ...` / `press ...` on the real machine.
    throttle_axis: StickAxis = StickAxis("left_y")
    steer_axis: StickAxis = StickAxis("left_x")
    turret_axis: Axis = StickAxis("right_x")
    arm_axis: Axis = ButtonAxis("boom_down", "boom_up")

    # --- calibration.  TODO(calibrate): all placeholders.
    stick_deadband: float = 0.10  # fraction of stick throw the toy ignores near centre
    drive_throttle: float = 0.6  # throttle fraction used while driving (0-1)
    drive_speed_mps: float = 0.10  # measured ground speed at drive_throttle
    steer_yaw_rate_deg_s: float = 30.0  # measured yaw rate at full steer while at drive_throttle
    turret_fraction: float = 1.0  # stick deflection while rotating the turret (0-1)
    turret_speed_deg_s: float = 20.0  # measured turret rate at turret_fraction
    arm_full_travel_s: float = 3.0  # seconds for the arm to travel its whole range

    # --- protocol
    keepalive_s: float = 0.2  # must be well under the firmware WATCHDOG_MS
    max_press_ms: int = 4000  # must be <= firmware MAX_PRESS_MS; longer presses are chunked

    def __post_init__(self):
        for f in ("drive_speed_mps", "steer_yaw_rate_deg_s", "turret_speed_deg_s", "arm_full_travel_s"):
            if getattr(self, f) <= 0:
                raise ValueError(f"{f} must be > 0")
        for f in ("stick_deadband", "drive_throttle", "turret_fraction"):
            if not 0 <= getattr(self, f) <= 1:
                raise ValueError(f"{f} must be within 0..1")

    # -- (de)serialisation ---------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("throttle_axis", "steer_axis", "turret_axis", "arm_axis"):
            d[k] = axis_to_dict(getattr(self, k))
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExcavatorConfig":
        base = cls()
        d = dict(d)
        for k in ("throttle_axis", "steer_axis", "turret_axis", "arm_axis"):
            if k in d:
                d[k] = axis_from_dict(d[k])
        unknown = set(d) - set(asdict(base))
        if unknown:
            raise ValueError(f"unknown config key(s): {sorted(unknown)}")
        return replace(base, **d)

    @classmethod
    def from_file(cls, path: str | Path) -> "ExcavatorConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
