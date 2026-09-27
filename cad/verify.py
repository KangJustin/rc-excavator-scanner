"""How close is the CAD model to the scan?  Distance from the scan's surface to the CAD surface."""
from __future__ import annotations

import numpy as np

from measure.fit import sample_surface


def to_cad_frame(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Scan coordinates -> CAD coordinates (pose takes CAD -> scan)."""
    T = np.linalg.inv(np.asarray(pose, float))
    return points @ T[:3, :3].T + T[:3, 3]


def deviation(scan_V: np.ndarray, scan_F: np.ndarray, pose, cad_stl, n: int = 20000, seed: int = 0) -> dict:
    """Unsigned distance from n points sampled evenly over the scan object's surface to the CAD surface (mm).
    Measured scan -> CAD: it answers 'how far does the scan stray from the model', so the scan's unseen
    underside cannot count against the model."""
    import open3d as o3d
    P, _ = sample_surface(scan_V, scan_F, n, np.random.default_rng(seed))
    P = to_cad_frame(P, pose)
    cad = o3d.io.read_triangle_mesh(str(cad_stl))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(cad))
    d = scene.compute_distance(o3d.core.Tensor(P.astype(np.float32))).numpy()
    size = float(np.linalg.norm(np.asarray(cad.vertices).max(axis=0) - np.asarray(cad.vertices).min(axis=0)))
    rms = float(np.sqrt(np.mean(d ** 2)))
    frac = rms / size if size else float("nan")
    grade = "excellent" if frac < 0.02 else "good" if frac < 0.05 else "rough" if frac < 0.10 else "poor"
    return {"rms_mm": rms, "mean_mm": float(d.mean()), "p95_mm": float(np.percentile(d, 95)),
            "max_mm": float(d.max()), "model_diagonal_mm": size, "rms_fraction_of_size": frac, "grade": grade,
            "points": int(n)}
