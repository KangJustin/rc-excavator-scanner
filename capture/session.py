"""A capture session: numbered images plus a CSV log, in one timestamped folder.

    captures/2026-09-25_2130/
        0001.jpg 0002.jpg ...
        captures.csv          one row per saved image (index, file, timestamp, ...)

The folder is created when the first image is saved, so starting the tool (or
pressing the new-session hotkey) and quitting never leaves empty folders behind.
"""
from __future__ import annotations

import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .camera import sharpness as measure_sharpness

CSV_FIELDS = ["index", "file", "timestamp", "unix_time", "width", "height", "sharpness", "extra"]
_IMAGE_RE = re.compile(r"^(\d{4,})\.jpg$")


def timestamp_name(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y-%m-%d_%H%M")


def unique_session_name(root: Path, now: datetime | None = None) -> str:
    """Timestamp name that doesn't collide with an existing folder
    (two sessions in the same minute get _2, _3, ...)."""
    base = timestamp_name(now)
    name, n = base, 1
    while (root / name).exists():
        n += 1
        name = f"{base}_{n}"
    return name


class CaptureSession:
    def __init__(self, root: str | Path = "captures", name: str | None = None,
                 jpeg_quality: int = 95):
        """name=None starts a brand-new, uniquely named session. An explicit name
        opens (and resumes numbering in) that folder if it already exists."""
        root = Path(root)
        self.name = name or unique_session_name(root)
        self.dir = root / self.name
        self.jpeg_quality = jpeg_quality

    @property
    def images_dir(self) -> Path:  # images live directly in the session folder
        return self.dir

    @property
    def log_path(self) -> Path:
        return self.dir / "captures.csv"

    metadata_path = log_path  # older name

    def _existing_indices(self) -> list[int]:
        if not self.dir.is_dir():
            return []
        return [int(m.group(1)) for p in self.dir.iterdir() if (m := _IMAGE_RE.match(p.name))]

    @property
    def count(self) -> int:
        return len(self._existing_indices())

    def save_frame(self, frame: np.ndarray, sharpness: float | None = None, **extra) -> Path:
        """Write the next numbered image (0001.jpg, 0002.jpg, ...) and append a log row.
        Extra keyword args (e.g. waypoint=3) are stored as JSON in the `extra` column.
        Numbering continues after the highest existing number, so deleting a bad shot
        never causes an overwrite."""
        self.dir.mkdir(parents=True, exist_ok=True)
        idx = max(self._existing_indices(), default=0) + 1
        path = self.dir / f"{idx:04d}.jpg"
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]):
            raise OSError(f"Failed to write {path}")
        h, w = frame.shape[:2]
        row = {
            "index": idx,
            "file": path.name,
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "unix_time": round(time.time(), 3),
            "width": w,
            "height": h,
            "sharpness": round(measure_sharpness(frame) if sharpness is None else sharpness, 1),
            "extra": json.dumps(extra) if extra else "",
        }
        new_file = not self.log_path.exists()
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, CSV_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        return path

    def write_info(self, info: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "session.json", "w") as f:
            json.dump(info, f, indent=2, default=str)

    def read_metadata(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        with open(self.log_path, newline="") as f:
            rows = []
            for r in csv.DictReader(f):
                extra = json.loads(r.pop("extra") or "{}")
                rec = {"index": int(r["index"]), "file": r["file"], "timestamp": r["timestamp"],
                       "unix_time": float(r["unix_time"]), "width": int(r["width"]),
                       "height": int(r["height"]), "sharpness": float(r["sharpness"])}
                rows.append({**rec, **extra})
            return rows
