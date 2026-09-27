"""Python-side excavator API.

`ExcavatorController` is the interface the rest of the project (orbit_capture) codes
against. `Excavator` implements it by talking to the ESP32 hardware bridge (or the
dry-run mock). Anything else that can drive the machine, e.g. a closed-loop /
learned controller later, can implement the same four methods and be swapped in.

The machine is open loop: there is no encoder or odometry, so motions are timed
from calibration constants in ExcavatorConfig. Expect drift; recalibrate often.
"""
from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from typing import Callable

from .config import (
    LINE_NAMES, STICK_MAX, STICK_MIN, STICK_NAMES, STICK_NEUTRAL,
    Axis, ButtonAxis, ExcavatorConfig, StickAxis,
)
from .transport import ControlError, DryRunTransport, SerialTransport, Transport, VirtualClock

log = logging.getLogger(__name__)


class ExcavatorController(ABC):
    @abstractmethod
    def drive_arc(self, radius: float, angle_deg: float) -> None:
        """Drive forward along a circular arc. radius in metres (> 0); angle_deg > 0 curves
        left / counter-clockwise, < 0 curves right / clockwise."""

    @abstractmethod
    def rotate_turret(self, angle_deg: float) -> None:
        """Rotate the cab relative to the tracks. > 0 = counter-clockwise seen from above."""

    @abstractmethod
    def move_arm(self, position: float) -> None:
        """Move the arm to a normalised position 0.0 .. 1.0."""

    @abstractmethod
    def press_button(self, name: str, duration_ms: float) -> None:
        """Hold a transmitter pushbutton down for duration_ms (blocks until done)."""

    @abstractmethod
    def stop(self) -> None:
        """Everything to neutral, all buttons released."""

    def close(self) -> None:
        self.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Excavator(ExcavatorController):
    def __init__(self, transport: Transport, config: ExcavatorConfig | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self._t = transport
        self.cfg = config or ExcavatorConfig()
        self._clock, self._sleep = clock, sleep
        self._arm_pos: float | None = None

    # ------------------------------------------------------------------ raw
    def set_stick(self, name: str, value: int) -> None:
        """Set one stick directly (0-255, 128 centred). Stays there until changed,
        NEUTRAL, or the firmware watchdog trips."""
        if name not in STICK_NAMES:
            raise ValueError(f"unknown stick {name!r}; expected one of {STICK_NAMES}")
        self._t.send(f"SET_STICK {name} {int(max(STICK_MIN, min(STICK_MAX, value)))}")

    def set_line(self, name: str, level: bool) -> None:
        """Drive one digital line (see control.config.LINE_NAMES) high or low. Stays there until
        changed, NEUTRAL, or the firmware watchdog trips."""
        if name not in LINE_NAMES:
            raise ValueError(f"unknown line {name!r}; expected one of {LINE_NAMES}")
        self._t.send(f"SET_LINE {name} {1 if level else 0}")

    def stop(self) -> None:
        self._t.send("NEUTRAL")

    def keepalive(self) -> None:
        """Reset the firmware watchdog (for callers that run their own control loop)."""
        self._t.send("KEEPALIVE")

    def press_button_nowait(self, name: str, duration_ms: float) -> None:
        """Start a button press and return immediately; the firmware releases it after
        duration_ms. Re-issue it every tick to hold a button from your own control loop."""
        self._t.send(f"PRESS_BUTTON {name} {int(max(1, min(self.cfg.max_press_ms, duration_ms)))}")

    def hold(self, seconds: float) -> None:
        """Keep the current stick positions for `seconds` (sending keepalives), then
        return everything to neutral."""
        self._run([], seconds)

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self._t.close()

    # --------------------------------------------------------------- motions
    def press_button(self, name: str, duration_ms: float) -> None:
        remaining = float(duration_ms)
        if remaining <= 0:
            raise ValueError("duration_ms must be > 0")
        while remaining > 0:
            chunk = min(remaining, self.cfg.max_press_ms)
            self._run([f"PRESS_BUTTON {name} {int(round(chunk))}"], chunk / 1000)
            remaining -= chunk

    def drive_arc(self, radius: float, angle_deg: float) -> None:
        if radius <= 0:
            raise ValueError("radius must be > 0 (metres)")
        if angle_deg == 0:
            return
        c = self.cfg
        speed = c.drive_speed_mps
        yaw_needed = speed / radius  # rad/s to stay on the circle
        steer = yaw_needed / math.radians(c.steer_yaw_rate_deg_s)
        if steer > 1.0:
            min_r = speed / math.radians(c.steer_yaw_rate_deg_s)
            raise ControlError(
                f"Radius {radius:.2f} m is tighter than this machine can drive at the calibrated "
                f"speed (minimum about {min_r:.2f} m). Use a larger radius or recalibrate.")
        seconds = radius * math.radians(abs(angle_deg)) / speed
        turn = 1.0 if angle_deg > 0 else -1.0  # + = left
        self._run([
            self._stick_cmd(c.throttle_axis, 1.0, c.drive_throttle),
            self._stick_cmd(c.steer_axis, turn, steer),
        ], seconds)

    def rotate_turret(self, angle_deg: float) -> None:
        if angle_deg == 0:
            return
        c = self.cfg
        self._move_axis(c.turret_axis, 1.0 if angle_deg > 0 else -1.0, c.turret_fraction,
                        abs(angle_deg) / c.turret_speed_deg_s)

    @property
    def dry_run_elapsed_s(self) -> float | None:
        """Simulated motion time so far (dry-run only; None on real hardware)."""
        return self._clock() if isinstance(self._t, DryRunTransport) else None

    @property
    def arm_position(self) -> float | None:
        """Dead-reckoned arm position, or None if unknown (call home_arm())."""
        return self._arm_pos

    def set_arm_estimate(self, position: float) -> None:
        """Tell the software where the arm currently is (no motion)."""
        self._arm_pos = float(position)

    def home_arm(self) -> None:
        """Drive the arm to its 0.0 end stop (overtravelling a little) and reset the estimate."""
        self._arm_pos = None
        self._move_axis(self.cfg.arm_axis, -1.0, 1.0, self.cfg.arm_full_travel_s * 1.2)
        self._arm_pos = 0.0

    def move_arm(self, position: float) -> None:
        if not 0.0 <= position <= 1.0:
            raise ValueError("position must be within 0.0 .. 1.0")
        if self._arm_pos is None:
            raise ControlError("Arm position unknown. Call home_arm() (drives to the 0.0 stop) "
                               "or set_arm_estimate(x) first.")
        delta = position - self._arm_pos
        if delta == 0:
            return
        self._arm_pos = None  # unknown if we get interrupted mid-move
        self._move_axis(self.cfg.arm_axis, 1.0 if delta > 0 else -1.0, 1.0,
                        abs(delta) * self.cfg.arm_full_travel_s)
        self._arm_pos = position

    # -------------------------------------------------------------- internals
    def _stick_value(self, fraction: float) -> int:
        """fraction in -1..1 -> 0..255 stick value, skipping the toy's dead zone."""
        if fraction == 0:
            return STICK_NEUTRAL
        d = self.cfg.stick_deadband
        mag = min(1.0, d + (1 - d) * abs(fraction))
        return int(max(STICK_MIN, min(STICK_MAX, round(STICK_NEUTRAL + math.copysign(mag, fraction) * 127))))

    def _stick_cmd(self, axis: StickAxis, direction: float, fraction: float) -> str:
        if axis.invert:
            direction = -direction
        return f"SET_STICK {axis.stick} {self._stick_value(direction * fraction)}"

    def _move_axis(self, axis: Axis, direction: float, fraction: float, seconds: float) -> None:
        if isinstance(axis, StickAxis):
            self._run([self._stick_cmd(axis, direction, fraction)], seconds)
        else:
            assert isinstance(axis, ButtonAxis)
            self.press_button(axis.plus if direction > 0 else axis.minus, seconds * 1000)

    def _run(self, commands: list[str], seconds: float) -> None:
        """Send commands, hold for `seconds` (with keepalives so the firmware watchdog
        stays quiet), then always return to neutral, even on Ctrl-C or error."""
        try:
            for cmd in commands:
                self._t.send(cmd)
            deadline = self._clock() + seconds
            while (left := deadline - self._clock()) > 0:
                self._sleep(min(self.cfg.keepalive_s, left))
                self._t.send("KEEPALIVE")
        finally:
            try:
                self.stop()
            except Exception:  # don't mask the original error
                log.exception("Failed to send NEUTRAL while cleaning up!")


def make_excavator(port: str | None = None, dry_run: bool = False,
                   config: ExcavatorConfig | None = None, baud: int = 115200,
                   show_keepalive: bool = False) -> Excavator:
    """Build an Excavator on real serial hardware, or on the print-only mock."""
    if dry_run:
        vc = VirtualClock()
        return Excavator(DryRunTransport(vc.now, show_keepalive=show_keepalive), config,
                         clock=vc.now, sleep=vc.sleep)
    if not port:
        raise ControlError("No serial port given. Pass --port (see `python -m control ports`) "
                           "or use --dry-run to run without hardware.")
    return Excavator(SerialTransport(port, baud), config)
