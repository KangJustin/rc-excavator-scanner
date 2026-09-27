"""Decide what to build, with which dimensions, and how the CAD model sits relative to the scan.

CAD frame: the model stands on Z = 0, centred on X/Y, so it is clean to edit and to print. `pose` is the
4x4 matrix taking CAD coordinates to the scan's coordinates (so a scan can be overlaid on the model).
"""
from __future__ import annotations

import numpy as np

SHAPES = ("sphere", "cylinder", "box", "solid")


def rotation_between(a, b) -> np.ndarray:
    """Rotation matrix taking unit vector a to unit vector b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0]) if abs(a[0]) < 0.9 else np.diag([-1.0, 1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def make_plan(res: dict, object_vertices: np.ndarray | None = None, force: str | None = None) -> dict:
    """res = measure.analyze() result. Returns {shape, params (mm), pose (4x4, CAD->scan), size_mm, fit}."""
    pf = res["primitive_fit"]
    z_up = res["object"]["z_is_up"]
    detected = pf["likely_shape"].split()[0]
    if detected == "flat" and not force:
        raise ValueError("No object found: the reconstruction contains only a flat surface (a floor or table). The object "
                         "probably failed to reconstruct (a hole in the mesh; shiny, transparent or plain objects often do). "
                         "Check preview.png and reshoot with a matte or textured object.")
    shape = force or ("solid" if detected == "neither" else detected)
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}")
    fit_note = f"{pf['likely_shape']}, fit {pf['fit_quality']}"
    z = np.array([0.0, 0.0, 1.0])

    if shape == "sphere":
        s = pf["sphere"]
        r = s["radius_mm"]
        params = {"DIAMETER_MM": 2 * r}
        pose = _pose(np.eye(3), np.array(s["center_mm"]) - r * z)
        size = 2 * r
    elif shape == "cylinder":
        c = pf["cylinder"]
        if c is None:
            raise ValueError("a cylinder could not be fitted to this object")
        axis, h = np.array(c["axis"]), c["height_mm"]
        params = {"DIAMETER_MM": 2 * c["radius_mm"], "HEIGHT_MM": h}
        R = rotation_between(z, axis)
        pose = _pose(R, np.array(c["center_mm"]) - R @ (z * h / 2))
        size = max(2 * c["radius_mm"], h)
    elif shape == "box":
        b = pf["box"]
        axes, dims = np.array(b["axes"]).T, np.array(b["dimensions_mm"])  # columns = axes
        iz = int(np.argmax(np.abs(axes[2]))) if z_up else int(np.argmin(dims))  # the vertical (or thinnest) side
        rest = sorted((i for i in range(3) if i != iz), key=lambda i: -dims[i])
        ix, iy = rest
        R = np.column_stack([axes[:, ix], axes[:, iy], axes[:, iz]])
        if np.linalg.det(R) < 0:
            R[:, 1] = -R[:, 1]
        L, W, H = dims[ix], dims[iy], dims[iz]
        params = {"LENGTH_MM": L, "WIDTH_MM": W, "HEIGHT_MM": H}
        pose = _pose(R, np.array(b["center_mm"]) - R @ (z * H / 2))
        size = max(L, W, H)
    else:  # solid: keep the scan's own geometry; stand it on Z=0, centred, when we know which way is up
        if object_vertices is None:
            raise ValueError("object vertices are required for the solid fallback")
        lo, hi = object_vertices.min(axis=0), object_vertices.max(axis=0)
        ctr = (lo + hi) / 2
        t = np.array([ctr[0], ctr[1], lo[2]]) if z_up else ctr
        params = {}
        pose = _pose(np.eye(3), t)
        size = float(np.linalg.norm(hi - lo))
    return {"shape": shape, "detected": detected, "fit_note": fit_note,
            "params": {k: round(float(v), 4) for k, v in params.items()},
            "pose": pose.tolist(), "size_mm": float(size), "z_up": bool(z_up)}
