"""Separate the object from the flat surface it sits on.

RANSAC (Open3D PointCloud.segment_plane on the mesh vertices) finds the dominant plane, but
"dominant" is not the same as "the table": on a box it can be the box's own top face. So a
candidate plane is only accepted if it passes sanity checks; otherwise the next-best plane is
tried, and after three misses we give up and the caller measures the whole mesh.

A plane is accepted only if ALL hold:
  1. it holds at least MIN_INLIER_FRACTION of the mesh vertices
  2. (if the mesh is known to be Z-up) it is horizontal, within MAX_TILT_DEG
  3. there is a connected object above it, resting on it (its lowest point is within a few
     tolerances of the plane)
  4. the plane is clearly wider than that object's footprint (MIN_EXTENT_RATIO): a support
     surface extends beyond what stands on it; a box's top face does not.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MIN_INLIER_FRACTION = 0.10
MAX_TILT_DEG = 20.0
MIN_EXTENT_RATIO = 1.5
MAX_ATTEMPTS = 3
CUT_FACTOR = 2.0  # vertices within CUT_FACTOR * tolerance of the plane count as "on" it


@dataclass
class Isolation:
    V: np.ndarray  # object vertices (compact)
    F: np.ndarray  # object triangles (indices into V)
    normal: np.ndarray | None = None  # unit normal of the base plane, pointing toward the object
    offset: float = 0.0  # height above the plane of point p is  p . normal + offset
    base_pts: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # object projected onto the plane (its footprint)
    info: dict = field(default_factory=dict)


def _o3d():
    import open3d as o3d
    return o3d


def _triangle_clusters(V, F):
    """[(area, triangle_index_array)] for each connected component, biggest first."""
    o3d = _o3d()
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    labels, counts, areas = m.cluster_connected_triangles()
    labels, areas = np.asarray(labels), np.asarray(areas)
    order = np.argsort(-areas)
    return [(float(areas[k]), np.where(labels == k)[0]) for k in order]


def _compact(V, F, tri_idx):
    used, inv = np.unique(F[tri_idx].ravel(), return_inverse=True)
    return V[used], inv.reshape(-1, 3)


def _footprint_diag(P, normal):
    """Diagonal of the 2D bounding box of P projected into the plane."""
    h = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(normal, h)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    xy = np.c_[P @ u, P @ v]
    return float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)))


def isolate_object(V: np.ndarray, F: np.ndarray, tol_frac: float = 0.008,
                   z_up: bool = False, enabled: bool = True) -> Isolation:
    """Return the object with the supporting plane removed, or the whole mesh (with the
    reason in info['reason']) if no trustworthy plane was found."""
    whole = Isolation(V, F)
    info = whole.info
    info.update(attempted=enabled, removed=False, reason=None, vertices_before=len(V))
    if not enabled:
        info["reason"] = "plane removal disabled (--no-plane-removal)"
        return whole
    if len(V) < 200:
        info["reason"] = "mesh too small for plane segmentation"
        return whole

    o3d = _o3d()
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    tol = tol_frac * diag
    info["tolerance"] = tol
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(V))
    remaining = np.arange(len(V))
    reasons = []

    for attempt in range(MAX_ATTEMPTS):
        if len(remaining) < max(100, MIN_INLIER_FRACTION * len(V)):
            break
        (a, b, c, d), inl = pcd.select_by_index(remaining).segment_plane(tol, 3, 3000)
        n = np.array([a, b, c])
        L = np.linalg.norm(n)
        n, d = n / L, d / L
        inl_g = remaining[inl]
        remaining = np.setdiff1d(remaining, inl_g)  # a rejected plane is excluded from the next try

        frac = len(inl_g) / len(V)
        if frac < MIN_INLIER_FRACTION:
            reasons.append(f"largest plane holds only {frac:.0%} of vertices (need {MIN_INLIER_FRACTION:.0%})")
            continue
        dist = V @ n + d
        if z_up:
            if n[2] < 0:
                n, d, dist = -n, -d, -dist
            tilt = float(np.degrees(np.arccos(np.clip(n[2], -1, 1))))
            if tilt > MAX_TILT_DEG:
                reasons.append(f"plane candidate is tilted {tilt:.0f} degrees from horizontal (likely a side of the object)")
                continue
        elif (dist > tol).sum() < (dist < -tol).sum():
            n, d, dist = -n, -d, -dist  # object side = the side with more geometry

        cut = CUT_FACTOR * tol
        keep_v = dist > cut
        tri_ok = keep_v[F].all(axis=1)
        if not tri_ok.any():
            reasons.append("nothing is left above the plane candidate")
            continue
        sub = np.where(tri_ok)[0]
        clusters = _triangle_clusters(V, F[sub])
        area, local = clusters[0]
        obj_tri = sub[local]
        OV, OF = _compact(V, F, obj_tri)
        h = OV @ n + d
        if h.min() > 6 * tol:
            reasons.append("the region above the plane candidate is floating, not resting on it")
            continue
        ext_plane = _footprint_diag(V[inl_g], n)
        ext_obj = _footprint_diag(OV, n)
        ratio = ext_plane / max(ext_obj, 1e-12)
        if ratio < MIN_EXTENT_RATIO:
            reasons.append(f"plane candidate is only {ratio:.1f}x wider than the object above it "
                           f"(need {MIN_EXTENT_RATIO}x); probably a face of the object, not a support surface")
            continue

        # Accepted. The object rests on the plane, so its true underside is ON the plane, even
        # though the camera never saw it and the cut removed a thin slab of its base. Project the
        # object onto the plane: this sets the base for bounding boxes / cylinder height without
        # changing the x/y footprint.
        base = OV - (h[:, None] * n)
        total_area = sum(a_ for a_, _ in clusters)
        info.update(removed=True, reason=None, plane_normal=n.tolist(), plane_offset=float(d),
                    plane_inlier_fraction=frac, plane_extent_ratio=float(ratio),
                    vertices_after=len(OV), attempts=attempt + 1, clusters=len(clusters),
                    object_area_share=float(area / total_area) if total_area else 1.0,
                    runner_up_area_ratio=float(clusters[1][0] / area) if len(clusters) > 1 else 0.0)
        if z_up:
            info["plane_tilt_deg"] = float(np.degrees(np.arccos(np.clip(n[2], -1, 1))))
        return Isolation(OV, OF, n, float(d), base, info)

    info["reason"] = ("no clean flat surface found: " + "; ".join(reasons)) if reasons else \
        "no plane candidates"
    return whole
