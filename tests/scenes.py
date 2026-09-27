"""Synthetic 'object on a table' scenes with known ground truth, meshed by the real
Poisson code, for testing measure/. Model units are arbitrary (UNIT_MM mm per unit)."""
import json
from pathlib import Path

import numpy as np

UNIT_MM = 50.0  # 1 model unit = 50 mm


def _cyl(r, h, n, rng):
    th, y = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, h, n)
    side = np.c_[r * np.cos(th), r * np.sin(th), y]
    sn = np.c_[np.cos(th), np.sin(th), 0 * th]
    m = int(n * 0.12)
    rr, t = r * np.sqrt(rng.uniform(0, 1, m)), rng.uniform(0, 2 * np.pi, m)
    top = np.c_[rr * np.cos(t), rr * np.sin(t), np.full(m, h)]
    return np.vstack([side, top]), np.vstack([sn, np.tile([0, 0, 1.0], (m, 1))])


def _sphere(r, n, rng):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v = v[v[:, 2] > -0.5]  # the part below the equator-ish is hidden by the table
    return v * r + [0, 0, r], v


def _box(a, b, c, n, rng):
    pts, nor = [], []
    for axis, sign, size in [(0, 1, (b, c)), (0, -1, (b, c)), (1, 1, (a, c)), (1, -1, (a, c)), (2, 1, (a, b))]:
        k = n // 5
        p = np.zeros((k, 3))
        dims = [a, b, c]
        other = [i for i in range(3) if i != axis]
        for i, j in enumerate(other):
            p[:, j] = rng.uniform(-dims[j] / 2, dims[j] / 2, k) + (dims[j] / 2 if j == 2 else 0)
        p[:, axis] = sign * dims[axis] / 2 + (dims[axis] / 2 if axis == 2 else 0) * (1 if sign > 0 else 0)
        nn = np.zeros((k, 3)); nn[:, axis] = sign
        pts.append(p); nor.append(nn)
    return np.vstack(pts), np.vstack(nor)


def make_scene(kind: str, tmp: Path, with_table=True, seed=0, noise=0.004, depth=8):
    """Returns (session_folder, truth) where truth is in mm."""
    import open3d as o3d
    from reconstruct.meshing import MeshOptions, build_mesh
    rng = np.random.default_rng(seed)
    if kind == "can":
        r, h = 0.66, 2.44  # 33 mm radius, 122 mm tall
        P, N = _cyl(r, h, 30000, rng)
        truth = dict(volume_mm3=np.pi * (r * UNIT_MM) ** 2 * h * UNIT_MM, height_mm=h * UNIT_MM, diameter_mm=2 * r * UNIT_MM)
    elif kind == "ball":
        r = 0.8
        P, N = _sphere(r, 30000, rng)
        truth = dict(volume_mm3=4 / 3 * np.pi * (r * UNIT_MM) ** 3, height_mm=2 * r * UNIT_MM, diameter_mm=2 * r * UNIT_MM)
    elif kind == "box":
        a, b, c = 1.2, 1.2, 1.6
        P, N = _box(a, b, c, 30000, rng)
        truth = dict(volume_mm3=a * b * c * UNIT_MM ** 3, height_mm=c * UNIT_MM)
    elif kind == "empty":  # a floor with nothing on it, e.g. the object failed to reconstruct
        P, N = np.zeros((0, 3)), np.zeros((0, 3))
        truth = {}
    elif kind == "tower":  # a block with a narrower tower on one end: irregular, no primitive describes it
        P1, N1 = _box(1.6, 0.8, 0.6, 20000, rng)
        P2, N2 = _box(0.6, 0.6, 1.4, 20000, rng)
        P2 = P2 + [-0.4, 0, 0]
        def inside(P, lo, hi):  # strictly inside an axis-aligned box
            return ((P > np.array(lo) + 1e-6) & (P < np.array(hi) - 1e-6)).all(axis=1)
        k1 = ~inside(P1, (-0.7, -0.3, 0), (-0.1, 0.3, 1.4))
        k2 = ~inside(P2, (-0.8, -0.4, 0), (0.8, 0.4, 0.6))
        P, N = np.vstack([P1[k1], P2[k2]]), np.vstack([N1[k1], N2[k2]])
        truth = dict(volume_mm3=(1.6 * 0.8 * 0.6 + 0.6 * 0.6 * 0.8) * UNIT_MM ** 3, height_mm=1.4 * UNIT_MM)
    else:
        raise ValueError(kind)
    if with_table:
        m = 30000
        T = np.c_[rng.uniform(-3, 3, (m, 2)), np.zeros(m)]
        P, N = np.vstack([P, T]), np.vstack([N, np.tile([0, 0, 1.0], (m, 1))])
    P = P + rng.normal(scale=noise, size=P.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    pcd.normals = o3d.utility.Vector3dVector(N)
    ws = tmp / f"{kind}_{'table' if with_table else 'alone'}"
    (ws / "dense").mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(ws / "dense" / "fused.ply"), pcd)
    res = build_mesh(ws / "dense" / "fused.ply", ws, MeshOptions(poisson_depth=depth, density_trim_quantile=0.02),
                     scene={"up": [0, 0, 1]})
    (ws / "report.json").write_text(json.dumps({"mesh": {"aligned": res.aligned}}))
    return ws, truth
