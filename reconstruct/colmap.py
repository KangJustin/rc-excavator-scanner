"""Running the COLMAP CLI, and reading COLMAP's own output files to diagnose problems.

Diagnosis reads the COLMAP database (SQLite) and the sparse model (binary) directly
instead of scraping logs, so it does not depend on log wording.
"""
from __future__ import annotations

import logging
import re
import shutil
import sqlite3
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .errors import ColmapCommandError, ColmapNotFoundError

log = logging.getLogger(__name__)

# COLMAP renamed the SIFT GPU flags in 4.x (SiftExtraction.* -> FeatureExtraction.*); we
# know both spellings, newest first, and fall back if one is rejected.
GPU_FLAGS = {
    "extract": ("FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"),
    "match": ("FeatureMatching.use_gpu", "SiftMatching.use_gpu"),
}

_GPU_TROUBLE = re.compile(r"(?i)cuda|opengl|glx|gpu|no display|could not create.*context")
_UNKNOWN_OPTION = re.compile(r"(?i)unrecognised option|unknown option|unrecognized option")


# ---------------------------------------------------------------------------
# Running external tools
# ---------------------------------------------------------------------------
class _NullFile:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def write(self, _): pass


def format_elapsed(seconds: float) -> str:
    """272 -> '04:32', 1127 -> '18:47', 3725 -> '1:02:05'."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


class _Heartbeat:
    """Daemon thread that prints '[stage] running... MM:SS elapsed' every `interval` seconds
    while a command runs. It only reads the clock and prints; it never touches the subprocess,
    so it can't affect the command or its exit code. interval <= 0 disables it."""

    def __init__(self, stage: str, interval: float):
        self.stage, self.interval = stage, interval
        self.start = time.monotonic()
        self._stop = threading.Event()
        self._thread = (threading.Thread(target=self._run, name=f"heartbeat-{stage}", daemon=True)
                        if interval > 0 else None)

    def __enter__(self) -> "_Heartbeat":
        if self._thread:
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def elapsed(self) -> str:
        return format_elapsed(time.monotonic() - self.start)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):  # wakes immediately when stopped
            print(f"[{self.stage}] running... {self.elapsed()} elapsed", flush=True)


def run_command(cmd: list[str], stage: str, log_dir: Path | None, verbose: bool = False,
                tool: str = "COLMAP", cwd: Path | None = None) -> str:
    """Run a command, stream its output to <log_dir>/<stage>.log, return the output.
    Raises ColmapCommandError on non-zero exit. While it runs, prints a heartbeat line to
    stdout every config.HEARTBEAT_SECONDS, then '[stage] done in MM:SS' (or 'failed after')."""
    log.info("[%s] %s", stage, " ".join(cmd))
    log_path = None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{stage}.log"
    lines: list[str] = []
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, cwd=cwd)
    except OSError as e:
        raise ColmapNotFoundError(f"Could not launch '{cmd[0]}': {e}") from e
    assert proc.stdout is not None
    with _Heartbeat(stage, config.HEARTBEAT_SECONDS) as beat:
        with (open(log_path, "w") if log_path else _NullFile()) as logf:
            for line in proc.stdout:
                lines.append(line)
                logf.write(line)
                if verbose:
                    print(line, end="")
        rc = proc.wait()
    output = "".join(lines)
    if rc != 0:
        print(f"[{stage}] failed after {beat.elapsed()}", flush=True)
        raise ColmapCommandError(stage, rc, output, log_path, tool)
    print(f"[{stage}] done in {beat.elapsed()}", flush=True)
    return output


@dataclass
class HelpProbe:
    ran: bool  # the binary launched and printed a COLMAP header (or exited 0)
    text: str  # raw `colmap help` output (or the launch error)
    version: str | None
    cuda: bool | None  # True = "with CUDA", False = "without CUDA", None = header doesn't say


@dataclass
class Colmap:
    binary: str = "colmap"
    log_dir: Path | None = None
    verbose: bool = False
    _probe: "HelpProbe | None" = field(default=None, repr=False)

    def ensure_available(self) -> None:
        if shutil.which(self.binary) is None:
            raise ColmapNotFoundError(
                f"COLMAP executable '{self.binary}' not found on PATH.",
                hint="Install COLMAP (see README) or point at it with --colmap-bin, the "
                     "COLMAP_BIN environment variable, or COLMAP_BIN in reconstruct/config.py.",
            )

    def probe(self) -> "HelpProbe":
        """Run `colmap help` once and parse its header, e.g.
        'COLMAP 3.9.1 -- ... (Commit abc on 2024-01-01 with CUDA)'."""
        if self._probe is None:
            try:
                out = subprocess.run([self.binary, "help"], capture_output=True, text=True, timeout=30)
                text, rc = out.stdout + out.stderr, out.returncode
            except (OSError, subprocess.TimeoutExpired) as e:
                text, rc = str(e), None
            m = re.search(r"COLMAP\s+(\d+(?:\.\d+)*\S*)", text)
            cuda = None
            if re.search(r"without CUDA", text):
                cuda = False
            elif re.search(r"with CUDA", text):
                cuda = True
            self._probe = HelpProbe(rc == 0 or m is not None, text, m.group(1) if m else None, cuda)
        return self._probe

    def version_and_cuda(self) -> tuple[str | None, bool | None]:
        """(version or None, has_cuda or None if the header doesn't say)."""
        p = self.probe()
        return p.version, p.cuda

    def run(self, stage: str, args: list[str]) -> str:
        """Run `colmap <args>`; return combined output. Raises ColmapCommandError."""
        return run_command([self.binary, *args], stage, self.log_dir, self.verbose)

    def run_gpu_aware(self, stage: str, kind: str, args: list[str], cpu_only: bool) -> str:
        """Run a SIFT stage. Honors cpu_only, and if the default (GPU) attempt
        fails for a GPU-looking reason, retries once on CPU."""
        names = GPU_FLAGS[kind]
        try:
            if not cpu_only:
                return self.run(stage, args)
            return self._run_cpu(stage, args, names)
        except ColmapCommandError as e:
            if not cpu_only and _GPU_TROUBLE.search(e.output):
                log.warning("[%s] failed with a GPU-related error; retrying on CPU.", stage)
                return self._run_cpu(stage, args, names)
            raise

    def _run_cpu(self, stage: str, args: list[str], names: tuple[str, str]) -> str:
        try:
            return self.run(stage, [*args, f"--{names[0]}", "0"])
        except ColmapCommandError as e:
            if _UNKNOWN_OPTION.search(e.output):
                return self.run(stage, [*args, f"--{names[1]}", "0"])
            raise


# ---------------------------------------------------------------------------
# Diagnosing the COLMAP database
# ---------------------------------------------------------------------------
_PAIR_MOD = 2147483647  # COLMAP packs (id1, id2) as id1 * 2147483647 + id2


@dataclass
class FeatureStats:
    per_image: dict[str, int]  # image name -> keypoint count

    @property
    def median(self) -> int:
        v = sorted(self.per_image.values())
        return v[len(v) // 2] if v else 0

    def below(self, threshold: int) -> list[str]:
        return sorted(n for n, c in self.per_image.items() if c < threshold)


@dataclass
class MatchStats:
    components: list[list[str]]  # groups of images connected by verified matches
    isolated: list[str]  # images with no verified match to any other image
    pair_count: int
    median_inliers: int

    @property
    def largest(self) -> int:
        return max((len(c) for c in self.components), default=0)


def read_feature_stats(db_path: Path) -> FeatureStats:
    con = sqlite3.connect(str(db_path))
    try:
        names = dict(con.execute("SELECT image_id, name FROM images"))
        counts = {i: r for i, r in con.execute("SELECT image_id, rows FROM keypoints")}
    finally:
        con.close()
    return FeatureStats({names[i]: counts.get(i, 0) for i in names})


def read_match_stats(db_path: Path, min_inliers: int = 15) -> MatchStats:
    """Build the image graph from geometrically verified pairs and find its
    connected components. `two_view_geometries.rows` = verified inlier matches."""
    con = sqlite3.connect(str(db_path))
    try:
        names = dict(con.execute("SELECT image_id, name FROM images"))
        rows = con.execute("SELECT pair_id, rows FROM two_view_geometries").fetchall()
    finally:
        con.close()

    parent = {i: i for i in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    connected, inliers = set(), []
    for pair_id, n in rows:
        if n < min_inliers:
            continue
        b = pair_id % _PAIR_MOD
        a = (pair_id - b) // _PAIR_MOD
        if a in parent and b in parent:
            parent[find(a)] = find(b)
            connected.update((a, b))
            inliers.append(n)

    groups: dict[int, list[str]] = {}
    for i in names:
        groups.setdefault(find(i), []).append(names[i])
    comps = sorted((sorted(g) for g in groups.values() if len(g) > 1), key=len, reverse=True)
    isolated = sorted(names[i] for i in names if i not in connected)
    inliers.sort()
    return MatchStats(comps, isolated, len(inliers), inliers[len(inliers) // 2] if inliers else 0)


# ---------------------------------------------------------------------------
# Reading the sparse model
# ---------------------------------------------------------------------------
@dataclass
class SparseModel:
    path: Path
    registered: list[str]  # names of images COLMAP could place
    num_points: int
    centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # camera centres, world frame
    up_dirs: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # each camera's "up" in world frame
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # sparse 3D points


def _quat_to_rot(q) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def read_sparse_model(model_dir: Path) -> SparseModel:
    """Parse images.bin / points3D.bin (COLMAP binary format): registered image
    names, camera centres and orientations, and the sparse 3D points."""
    names, centers, ups = [], [], []
    with open(model_dir / "images.bin", "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            # image_id, qvec (w,x,y,z), tvec, camera_id
            _id, qw, qx, qy, qz, tx, ty, tz, _cam = struct.unpack("<i4d3di", f.read(64))
            R = _quat_to_rot(np.array([qw, qx, qy, qz]))  # world -> camera
            centers.append(-R.T @ np.array([tx, ty, tz]))
            ups.append(-R[1])  # camera y axis points down, so up = -(row 1 of R) in world coordinates
            name = bytearray()
            while (c := f.read(1)) not in (b"\x00", b""):
                name += c
            names.append(name.decode())
            (n2d,) = struct.unpack("<Q", f.read(8))
            f.seek(n2d * 24, 1)  # (x, y, point3D_id) per 2D point
    pts = []
    with open(model_dir / "points3D.bin", "rb") as f:
        (npts,) = struct.unpack("<Q", f.read(8))
        rec = struct.Struct("<Q3d3BdQ")  # id, xyz, rgb, error, track length
        for _ in range(npts):
            r = rec.unpack(f.read(rec.size))
            pts.append(r[1:4])
            f.seek(r[-1] * 8, 1)  # track: (image_id, point2D_idx) int32 pairs
    return SparseModel(model_dir, names, npts, np.array(centers).reshape(-1, 3),
                       np.array(ups).reshape(-1, 3), np.array(pts).reshape(-1, 3))


def find_best_model(sparse_dir: Path) -> SparseModel | None:
    """The mapper writes sparse/0, sparse/1, ... (one per disconnected
    component). Return the one with the most registered images."""
    models = []
    for d in sorted(p for p in sparse_dir.glob("*") if p.is_dir()):
        if (d / "images.bin").exists() and (d / "points3D.bin").exists():
            models.append(read_sparse_model(d))
    return max(models, key=lambda m: len(m.registered), default=None)


def count_models(sparse_dir: Path) -> int:
    return sum(1 for d in sparse_dir.glob("*") if (d / "images.bin").exists())


# ---------------------------------------------------------------------------
# PLY header
# ---------------------------------------------------------------------------
def ply_vertex_count(path: Path) -> int:
    with open(path, "rb") as f:
        head = f.read(4096).decode("latin-1")
    m = re.search(r"element vertex (\d+)", head)
    return int(m.group(1)) if m else 0
