"""Sphere and cylinder fitting on surface samples.

Fits use points sampled uniformly by AREA from the triangle mesh (not raw vertices,
whose density varies), so a densely-meshed cap can't outvote a sparsely-meshed side.
Residuals are true geometric distances to the fitted shape.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sample_surface(V: np.ndarray, F: np.ndarray, n: int, rng: np.random.Generator):
    """n points uniformly distributed over the mesh surface, with their face normals."""
    tri = V[F]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    cr = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    ok = area > 1e-15
    tri, e1, e2, cr, area = tri[ok], e1[ok], e2[ok], cr[ok], area[ok]
    normals = cr / (2 * area[:, None])
    idx = rng.choice(len(area), n, p=area / area.sum())
    u, v = rng.random((n, 1)), rng.random((n, 1))
    flip = (u + v) > 1
    u, v = np.where(flip, 1 - u, u), np.where(flip, 1 - v, v)
    return tri[idx, 0] + u * e1[idx] + v * e2[idx], normals[idx]


@dataclass
class SphereFit:
    center: np.ndarray
    radius: float
    rms: float  # RMS of |p - c| - r
    p95: float  # 95th percentile of |residual|

    @property
    def rel_rms(self) -> float:
        return self.rms / self.radius


def fit_sphere(P: np.ndarray) -> SphereFit:
    """Linear least squares: |p|^2 = 2 c.p + (r^2 - |c|^2)."""
    m = P.mean(axis=0)
    Q = P - m
    A = np.c_[2 * Q, np.ones(len(Q))]
    x, *_ = np.linalg.lstsq(A, (Q ** 2).sum(axis=1), rcond=None)
    c = x[:3]
    r = float(np.sqrt(max(x[3] + c @ c, 1e-18)))
    res = np.linalg.norm(Q - c, axis=1) - r
    return SphereFit(c + m, r, float(np.sqrt(np.mean(res ** 2))), float(np.percentile(np.abs(res), 95)))


@dataclass
class CylinderFit:
    axis: np.ndarray  # unit vector
    center: np.ndarray  # midpoint of the axis segment
    radius: float
    height: float
    rms: float  # RMS distance from every point to the finite, capped cylinder surface
    p95: float
    side_fraction: float  # share of surface points whose normal is perpendicular to the axis

    @property
    def rel_rms(self) -> float:
        return self.rms / self.radius


def _basis(a: np.ndarray):
    h = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, h)
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


def _fit_circle(x: np.ndarray, y: np.ndarray):
    """Kasa algebraic circle fit."""
    mx, my = x.mean(), y.mean()
    xc, yc = x - mx, y - my
    A = np.c_[2 * xc, 2 * yc, np.ones(len(xc))]
    s, *_ = np.linalg.lstsq(A, xc ** 2 + yc ** 2, rcond=None)
    return s[0] + mx, s[1] + my, float(np.sqrt(max(s[2] + s[0] ** 2 + s[1] ** 2, 1e-18)))


def _distance_to_capped_cylinder(rho, t, r, t0, t1):
    """Exact distance from points (radial distance rho, axial coordinate t) to the closed
    surface: lateral wall + two end discs."""
    lateral = np.hypot(np.abs(rho - r), np.maximum(0, np.maximum(t0 - t, t - t1)))
    cap0 = np.hypot(np.maximum(0, rho - r), t - t0)
    cap1 = np.hypot(np.maximum(0, rho - r), t - t1)
    return np.minimum(lateral, np.minimum(cap0, cap1))


def _evaluate_axis(P, N, a, extent_pts):
    u, v = _basis(a)
    side = np.abs(N @ a) < 0.3
    if side.sum() < max(50, 0.03 * len(P)):
        return None
    x, y = P @ u, P @ v
    xs, ys = x[side], y[side]
    keep = np.ones(len(xs), bool)
    for _ in range(3):  # re-fit on inliers only, so caps/bumps don't drag the circle
        cx, cy, r = _fit_circle(xs[keep], ys[keep])
        res = np.abs(np.hypot(xs - cx, ys - cy) - r)
        s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        new_keep = res < max(3 * s, 0.005 * r)
        if new_keep.sum() < 20:
            break
        keep = new_keep
    t_all = extent_pts @ a
    t0, t1 = float(t_all.min()), float(t_all.max())
    rho = np.hypot(x - cx, y - cy)
    d = _distance_to_capped_cylinder(rho, P @ a, r, t0, t1)
    center = cx * u + cy * v + 0.5 * (t0 + t1) * a
    return CylinderFit(a, center, r, t1 - t0, float(np.sqrt(np.mean(d ** 2))),
                       float(np.percentile(d, 95)), float(side.mean()))


def _refine_axis(N, a, iters=3):
    """A cylinder's side normals are all perpendicular to its axis: the axis is the
    direction of least normal variance among the side-like normals."""
    for _ in range(iters):
        side = np.abs(N @ a) < 0.15
        if side.sum() < 30:
            break
        w, vec = np.linalg.eigh(N[side].T @ N[side])
        a = vec[:, 0]
    return a


def fit_cylinder(P: np.ndarray, N: np.ndarray, extent_pts: np.ndarray | None = None,
                 rng: np.random.Generator | None = None, ransac_iters: int = 400) -> CylinderFit | None:
    """Axis candidates come from (1) RANSAC over pairs of surface normals (two normals on a
    cylinder wall span a plane perpendicular to the axis, so a = n1 x n2) and (2) the PCA
    axes. Each candidate is scored by the actual geometric residual; the best wins."""
    rng = rng or np.random.default_rng(0)
    extent_pts = P if extent_pts is None else extent_pts
    n = len(P)
    i, j = rng.integers(0, n, ransac_iters), rng.integers(0, n, ransac_iters)
    cand = np.cross(N[i], N[j])
    norm = np.linalg.norm(cand, axis=1)
    good = norm > 0.3
    cand = cand[good] / norm[good, None]
    candidates = []
    if len(cand):
        scores = np.array([np.mean(np.abs(N @ a) < 0.08) for a in cand])
        picked = []
        for k in np.argsort(-scores):
            a = cand[k]
            if all(abs(a @ b) < 0.98 for b in picked):
                picked.append(a)
            if len(picked) == 3:
                break
        candidates += [_refine_axis(N, a) for a in picked]
    _, _, Vt = np.linalg.svd(P - P.mean(axis=0), full_matrices=False)
    candidates += list(Vt)

    best = None
    for a in candidates:
        a = a / np.linalg.norm(a)
        fit = _evaluate_axis(P, N, a, extent_pts)
        if fit and (best is None or fit.rms < best.rms):
            best = fit
    if best is not None and best.axis[np.argmax(np.abs(best.axis))] < 0:
        best.axis = -best.axis  # canonical sign
    return best


@dataclass
class BoxFit:
    center: np.ndarray  # world position of the box centre
    axes: np.ndarray  # 3x3, columns = the box's axes (unit, right-handed)
    dims: np.ndarray  # side lengths along each axis column
    rms: float  # RMS distance from the surface samples to the box surface
    p95: float

    @property
    def rel_rms(self) -> float:
        return self.rms / (self.dims.min() / 2)


def fit_box(P: np.ndarray, extent_pts: np.ndarray, R: np.ndarray) -> BoxFit:
    """Box aligned to the given axes R (columns), sized to enclose extent_pts. Residual = distance
    from each surface sample to the nearest face (inside or outside the box)."""
    q_all = extent_pts @ R
    lo, hi = q_all.min(axis=0), q_all.max(axis=0)
    dims, cq = hi - lo, (hi + lo) / 2
    q, h = P @ R - cq, dims / 2
    outside = np.linalg.norm(np.maximum(np.abs(q) - h, 0), axis=1)
    inside = (h - np.abs(q)).min(axis=1)
    d = np.where((np.abs(q) <= h).all(axis=1), inside, outside)
    return BoxFit(R @ cq, R, dims, float(np.sqrt(np.mean(d ** 2))), float(np.percentile(d, 95)))
