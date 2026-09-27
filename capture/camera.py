"""Webcam access (Logitech C270 over USB) via OpenCV."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    pass


def sharpness(frame: np.ndarray) -> float:
    """Variance of the Laplacian. Higher = sharper. Only comparable between
    frames of the same scene/camera; there is no universal 'good' number."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


MIN_BRIGHTNESS = 90  # mean grey (0-255) below which photos are too dark: noisy, poor matching, muddy colours


def light_status(frame: np.ndarray) -> tuple[float, bool]:
    """(mean brightness 0-255, is it bright enough?)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    b = float(gray.mean())
    return b, b >= MIN_BRIGHTNESS


@dataclass
class FrameStats:
    width: int
    height: int
    sharpness: float
    mean_brightness: float  # 0-255
    clipped_pct: float  # % of pixels at (near) black or white


def frame_stats(frame: np.ndarray) -> FrameStats:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clipped = float(np.mean((gray <= 3) | (gray >= 252)) * 100)
    h, w = gray.shape
    return FrameStats(w, h, sharpness(frame), float(gray.mean()), clipped)


class Camera:
    """Thin wrapper around cv2.VideoCapture.

    USB webcams buffer a few frames in the driver, so a frame grabbed after the
    camera has sat idle (e.g. while the excavator was driving) can be stale.
    `read_fresh` discards the buffer first; `grab_best` additionally picks the
    sharpest of several fresh frames, which is cheap insurance against motion
    blur right after a move.
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        warmup_frames: int = 20,
        flush_frames: int = 5,
        lock_exposure: bool = False,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_frames = warmup_frames
        self.flush_frames = flush_frames
        self.lock_exposure = lock_exposure
        self._cap: cv2.VideoCapture | None = None

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> "Camera":
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise CameraError(
                f"Could not open camera index {self.index}. Try `python -m capture "
                "--list-cameras`. On macOS, grant camera access to your terminal "
                "(System Settings > Privacy & Security > Camera); on Linux, check "
                "you are in the `video` group."
            )
        # MJPG is needed to get 720p at 30 fps over USB2; raw YUYV drops to ~10 fps.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ignored by some backends
        self._cap = cap

        # Auto exposure / white balance need a moment to settle.
        for _ in range(self.warmup_frames):
            cap.read()
        frame = self.read()  # raises if the camera gives nothing

        actual = (frame.shape[1], frame.shape[0])
        if actual != (self.width, self.height):
            log.warning(
                "Requested %dx%d but camera delivers %dx%d; using the latter.",
                self.width, self.height, *actual,
            )
            self.width, self.height = actual
        if self.lock_exposure:
            self._lock_exposure()
        return self

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- frames ------------------------------------------------------------
    def read(self) -> np.ndarray:
        if self._cap is None:
            raise CameraError("Camera is not open.")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraError(
                "Camera returned no frame (unplugged, or in use by another app?)."
            )
        return frame

    def read_fresh(self) -> np.ndarray:
        for _ in range(self.flush_frames):
            self.read()
        return self.read()

    def grab_best(self, n: int = 5) -> tuple[np.ndarray, float]:
        """Return (sharpest of n fresh frames, its sharpness)."""
        for _ in range(self.flush_frames):
            self.read()
        best, best_s = None, -1.0
        for _ in range(max(1, n)):
            f = self.read()
            s = sharpness(f)
            if s > best_s:
                best, best_s = f, s
        return best, best_s

    # -- helpers -----------------------------------------------------------
    def _lock_exposure(self) -> None:
        """Best effort: consistent exposure/white balance across the orbit helps
        SfM feature matching. Support depends on OS backend, and OpenCV gives no
        reliable way to confirm it took effect."""
        cap = self._cap
        exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
        # V4L2 uses 1 = manual; other backends use 0.25 for manual.
        manual = 1 if cap.getBackendName() == "V4L2" else 0.25
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        log.info(
            "Requested exposure/white-balance lock (exposure=%s). Verify in the preview "
            "that brightness no longer drifts; some backends ignore this.", exposure,
        )


class FakeCamera:
    """Synthetic camera for testing the pipeline without hardware.

    Frames are textured noise with the frame number burned in. They are NOT
    suitable for real reconstruction; they exist so orbit_capture can be
    exercised end to end on a laptop.
    """

    def __init__(self, width: int = 640, height: int = 480, seed: int = 0):
        self.width, self.height = width, height
        self._rng = np.random.default_rng(seed)
        self._n = 0

    def open(self) -> "FakeCamera":
        return self

    def close(self) -> None:
        pass

    def __enter__(self) -> "FakeCamera":
        return self

    def __exit__(self, *exc) -> None:
        pass

    def read(self) -> np.ndarray:
        self._n += 1
        img = self._rng.integers(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (0, 0), 3)
        cv2.putText(img, f"fake {self._n}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        return img

    read_fresh = read

    def grab_best(self, n: int = 5) -> tuple[np.ndarray, float]:
        f = self.read()
        return f, sharpness(f)


def list_cameras(max_index: int = 6, snapshot_dir: str | Path | None = None,
                 warmup_frames: int = 10) -> list[tuple[int, int, int, Path | None]]:
    """Probe indices 0..max_index-1; return [(index, width, height, snapshot_path)] for each
    that opens. OpenCV can't give device names, so if snapshot_dir is set a JPEG from each
    camera is saved there: look at them to tell the built-in camera from the USB one."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            frame = None
            for _ in range(max(1, warmup_frames)):  # first frames are often black while exposure settles
                ok, f = cap.read()
                if ok and f is not None:
                    frame = f
            if frame is not None:
                snap = None
                if snapshot_dir is not None:
                    Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
                    snap = Path(snapshot_dir) / f"camera_{i}.jpg"
                    cv2.imwrite(str(snap), frame)
                found.append((i, frame.shape[1], frame.shape[0], snap))
        cap.release()
    return found


def measure_fps(cam: Camera, seconds: float = 2.0) -> float:
    n, t0 = 0, time.monotonic()
    while time.monotonic() - t0 < seconds:
        cam.read()
        n += 1
    return n / (time.monotonic() - t0)
