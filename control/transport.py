"""Line-based serial protocol to the ESP32, plus a print-only mock.

Wire format (ASCII, one command per line, '\\n' terminated, 115200 baud):

    PING                        -> OK PONG
    KEEPALIVE                   -> OK        (resets the firmware watchdog)
    NEUTRAL                     -> OK        (all sticks to 128, all buttons released, all lines low)
    SET_STICK <name> <0-255>    -> OK        (128 = centred)
    SET_LINE <name> <0|1>       -> OK        (digital line low / high; names: control.config.LINE_NAMES)
    PRESS_BUTTON <name> <ms>    -> OK        (returns immediately; firmware releases after <ms>)
    STATUS                      -> OK <stick=value ...> <button=0|1 ...>

Errors come back as `ERR <reason>`. Any other line the firmware prints (READY,
WARN ...) is logged and skipped.
"""
from __future__ import annotations

import logging
import sys
import time
from abc import ABC, abstractmethod
from typing import Callable

log = logging.getLogger(__name__)


class ControlError(RuntimeError):
    pass


class Transport(ABC):
    @abstractmethod
    def send(self, command: str) -> str:
        """Send one command; return the text after 'OK' (may be empty).
        Raises ControlError on ERR / timeout."""

    def close(self) -> None:
        pass


class SerialTransport(Transport):
    def __init__(self, port: str, baud: int = 115200, timeout: float = 1.0,
                 boot_wait: float = 2.0):
        try:
            import serial
        except ImportError as e:
            raise ControlError("pyserial is not installed (pip install pyserial).") from e
        try:
            self._ser = serial.Serial(port, baud, timeout=timeout)
        except serial.SerialException as e:
            raise ControlError(
                f"Could not open serial port {port}: {e}. Check `python -m control ports`, "
                "that nothing else (Arduino serial monitor) has it open, and on Linux that "
                "you are in the `dialout` group."
            ) from e
        self._timeout = timeout
        time.sleep(boot_wait)  # opening the port resets most ESP32 boards
        self._ser.reset_input_buffer()
        self.send("PING")

    def send(self, command: str) -> str:
        self._ser.write((command + "\n").encode("ascii"))
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            line = self._ser.readline().decode("ascii", errors="replace").strip()
            if not line:
                continue
            if line == "OK" or line.startswith("OK "):
                return line[3:]
            if line.startswith("ERR"):
                raise ControlError(f"ESP32 rejected {command!r}: {line}")
            log.info("esp32: %s", line)
        raise ControlError(f"No reply from ESP32 to {command!r} within {self._timeout}s "
                           "(wrong port/baud, or firmware not flashed?).")

    def close(self) -> None:
        try:
            self._ser.write(b"NEUTRAL\n")
        finally:
            self._ser.close()


class VirtualClock:
    """A clock that only advances when slept on, so dry runs are instant but still
    report how long the motion would take."""

    def __init__(self):
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class DryRunTransport(Transport):
    """Prints commands instead of sending them. No hardware needed."""

    def __init__(self, clock: Callable[[], float] | None = None,
                 out: Callable[[str], None] | None = None, show_keepalive: bool = False):
        self._clock = clock or (lambda: 0.0)
        self._out = out or (lambda s: print(s, file=sys.stdout))
        self._show_keepalive = show_keepalive
        self.sent: list[str] = []

    def send(self, command: str) -> str:
        self.sent.append(command)
        if command != "KEEPALIVE" or self._show_keepalive:
            self._out(f"[dry-run t={self._clock():7.2f}s] {command}")
        return "PONG" if command == "PING" else ""
