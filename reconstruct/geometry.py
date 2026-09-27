"""Is the solved camera path good enough to reconstruct an object?

Works on the sparse model's camera centres and 3D points, so it judges what COLMAP
actually solved (registered cameras only), not what you intended to shoot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CoverageReport:
    n_cameras: int
    linearity: float  # 2nd / 1st principal spread of camera centres: ~0 = a straight line
    coverage_deg: float  # angular extent of the cameras around the object
    largest_gap_deg: float
    gap_between: tuple[str, str] | None  # the two images either side of the biggest gap
    elevation_span_deg: float  # spread of camera heights as seen from the object
    up: np.ndarray  # world-space "up" (normal of the camera orbit plane)
    center: np.ndarray  # estimated object centre (world)


def analyze_camera_coverage(names: list[str], centers: np.ndarray, up_dirs: np.ndarray,
                            points: np.ndarray) -> CoverageReport:
    C = np.asarray(centers, float)
    m = C.mean(axis=0)
    _, S, Vt = np.linalg.svd(C - m, full_matrices=False)
    linearity = float(S[1] / S[0]) if S[0] > 1e-12 else 0.0

    # The orbit plane's normal is "up"; its sign is fixed by which way the cameras' own up points.
    normal = Vt[2]
    if len(up_dirs) and float(np.mean(up_dirs, axis=0) @ normal) < 0:
        normal = -normal

    obj = np.median(points, axis=0) if len(points) else m
    p = (C - obj) @ Vt[:2].T  # camera positions around the object, projected into the orbit plane
    ang = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
    order = np.argsort(ang)
    a = ang[order]
    gaps = np.diff(np.append(a, a[0] + 360.0))
    k = int(np.argmax(gaps))
    largest = float(gaps[k])
    between = (names[order[k]], names[order[(k + 1) % len(a)]])

    height = (C - m) @ normal
    elev = np.degrees(np.arctan2(height, np.linalg.norm(p, axis=1)))
    return CoverageReport(len(C), linearity, 360.0 - largest, largest, between,
                          float(elev.max() - elev.min()), normal, obj)
