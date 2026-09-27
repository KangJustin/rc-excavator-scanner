"""Mesh -> real-world dimensions, volume, surface area and primitive-shape fits (all in mm)."""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np

from .fit import fit_box, fit_cylinder, fit_sphere, sample_surface
from .plane import Isolation, isolate_object

# Relative fit error = RMS residual / radius. Photogrammetry noise is typically a few percent.
GOOD_REL_RMS = 0.04
FLAT_RATIO = 0.03  # thinnest / widest side of the whole mesh below this = a flat sheet, not an object
ROUGH_REL_RMS = 0.10


class MeasureError(Exception):
    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.message, self.hint = message, hint

    def __str__(self) -> str:
        return self.message + (f"\n  What to try: {self.hint}" if self.hint else "")


def load_mesh(path: Path):
    """Read a mesh and weld duplicate vertices. STL stores every triangle separately, so
    without welding no two triangles share a vertex and connectivity-based steps fail."""
    import open3d as o3d
    if not path.is_file():
        raise MeasureError(f"Mesh not found: {path}",
                           hint="Run `python -m reconstruct captures/<session>` first; it writes mesh.stl.")
    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.triangles, int)
    if len(F) < 20:
        raise MeasureError(f"{path.name} has only {len(F)} triangles; nothing to measure.",
                           hint="The reconstruction probably failed; check its preview.png.")
    return V, F


def _r(x, nd=None):
    """Round to 5 significant figures (so precision survives whichever units the mesh is in,
    e.g. raw model units of ~1 as well as millimetres of ~100,000). `nd` is accepted and ignored."""
    return None if x is None else float(f"{float(x):.5g}")


def _vec(v, nd=None):
    return [_r(t) for t in v]


def _triangles(V, F):
    tri = V[F]
    return tri, np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])


def surface_area(V, F) -> float:
    return float(0.5 * np.linalg.norm(_triangles(V, F)[1], axis=1).sum())


def volume_over_plane(V, F, normal, offset) -> float:
    """Volume between the surface and the base plane (divergence theorem with prisms):
    each triangle contributes (its area projected onto the plane, signed by orientation)
    x (its mean height). Works for a mesh that is open at the bottom because the plane
    closes it. Vertical walls project to ~0 so they add nothing."""
    tri, cr = _triangles(V, F)
    h = (tri @ normal + offset).mean(axis=1)
    return float(0.5 * (cr @ normal) @ h)


def volume_closed(V, F) -> float:
    """Signed volume of a closed mesh by summing tetrahedra to the centroid."""
    tri, _ = _triangles(V - V.mean(axis=0), F)
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def _is_closed(V, F) -> bool:
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    return bool(m.is_edge_manifold(False) and m.is_vertex_manifold())


def _pca_box(P):
    """Oriented box along the principal axes (plain numpy, always valid)."""
    _, vec = np.linalg.eigh(np.cov((P - P.mean(axis=0)).T))
    if np.linalg.det(vec) < 0:
        vec[:, 0] = -vec[:, 0]
    q = (P - P.mean(axis=0)) @ vec
    return q.max(axis=0) - q.min(axis=0), vec


def _oriented_box(P):
    """(extents sorted longest->shortest, box axes as matrix columns). Uses Open3D's minimal
    oriented box when it returns a valid rotation, else a PCA box."""
    import open3d as o3d
    ext, R = None, None
    try:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            obb = pcd.get_minimal_oriented_bounding_box()
        e, r = np.asarray(obb.extent), np.asarray(obb.R)
        if np.all(np.isfinite(e)) and np.allclose(r.T @ r, np.eye(3), atol=1e-5) and np.linalg.det(r) > 0:
            ext, R = e, r
    except Exception:
        pass
    if ext is None:
        ext, R = _pca_box(P)
    order = np.argsort(-ext)
    return ext[order], R[:, order]


def _hull_volume(P):
    import open3d as o3d
    try:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        hull, _ = pcd.compute_convex_hull()
        return float(hull.get_volume())
    except Exception:
        return None


def analyze(mesh_path: Path, scale_mm: float | None = None, plane_removal: bool = True,
            plane_tol: float = 0.008, samples: int = 20000, z_up: bool = False,
            seed: int = 0, return_object: bool = False, scale_info: dict | None = None):
    """Measure a mesh. scale_mm = millimetres per model unit (None = unscaled, reported as-is).
    With return_object=True returns (result, object_vertices, object_triangles): the isolated,
    scaled object mesh, which the cad/ module compares its model against."""
    from .report import format_summary
    S = 1.0 if scale_mm is None else float(scale_mm)
    if not S > 0:  # validate arguments before doing any work
        raise MeasureError("--scale-mm must be positive (millimetres per model unit).")
    V, F = load_mesh(Path(mesh_path))
    V = V * S
    warnings: list[str] = []
    if scale_mm is None:
        warnings.append("No --scale-mm given: numbers are in raw model units, NOT real millimetres. "
                        "Photogrammetry has arbitrary scale; measure one real dimension of the object "
                        "and rerun with --scale-mm (mm per model unit).")

    iso: Isolation = isolate_object(V, F, plane_tol, z_up, plane_removal)
    OV, OF = iso.V, iso.F
    have_plane = iso.normal is not None
    if plane_removal and not have_plane:
        warnings.append(f"Plane removal skipped, measured the whole mesh: {iso.info['reason']}. "
                        "If a table or background is in the mesh, these numbers include it.")
    if have_plane and iso.info.get("runner_up_area_ratio", 0) > 0.5:
        warnings.append("Another large piece remained after removing the plane "
                        f"({iso.info['runner_up_area_ratio']:.0%} the size of the chosen object); "
                        "the mesh may contain more than one object or leftover background.")

    # Points that describe the object's true extent: its vertices plus the base it stands on.
    extent_pts = np.vstack([OV, iso.base_pts]) if have_plane and len(iso.base_pts) else OV
    lo, hi = extent_pts.min(axis=0), extent_pts.max(axis=0)
    obb_ext, obb_R = _oriented_box(extent_pts)
    height_above = float((OV @ iso.normal + iso.offset).max()) if have_plane else None

    # Volume ---------------------------------------------------------------------------------
    hull_vol = _hull_volume(extent_pts)
    vol, method, vol_note = None, None, None
    if have_plane:
        vol = abs(volume_over_plane(OV, OF, iso.normal, iso.offset))
        method, vol_note = "surface-to-base-plane", "Volume between the object's surface and the base plane it sits on."
    elif _is_closed(OV, OF):
        vol = abs(volume_closed(OV, OF))
        method, vol_note = "closed-mesh", "Mesh is closed; exact enclosed volume."
    else:
        vol_note = ("Mesh is open (not watertight) and no base plane was found, so a volume would be "
                    "unreliable. Use the convex-hull volume as an upper bound.")
        warnings.append(vol_note)
    area = surface_area(OV, OF)

    # Primitive fits -----------------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    P, N = sample_surface(OV, OF, samples, rng)
    sph = fit_sphere(P)
    cyl = fit_cylinder(P, N, extent_pts, rng)
    box = fit_box(P, extent_pts, obb_R)
    cands = [("sphere", sph.rel_rms), ("box", box.rel_rms)] + ([("cylinder", cyl.rel_rms)] if cyl else [])
    best, rel = min(cands, key=lambda t: t[1])
    quality = "good" if rel < GOOD_REL_RMS else "rough" if rel < ROUGH_REL_RMS else "poor"
    likely = best if quality != "poor" else "neither (irregular or a composite shape)"
    flat_ratio = float(obb_ext[2] / obb_ext[0]) if obb_ext[0] else 1.0
    if not have_plane and flat_ratio < FLAT_RATIO:
        # No table was found to remove AND the whole mesh is a thin sheet: this is a floor/table with nothing standing
        # on it. A very wide, very short "cylinder" fits a flat sheet perfectly, so it must not be reported as one.
        likely, quality = "flat surface (nothing standing on it)", "poor"
        warnings.append(
            f"The mesh is a flat sheet (thickness is {flat_ratio:.1%} of its width): a floor or table with nothing standing "
            "on it. The object is probably MISSING from the reconstruction (a hole where it should be). Shiny, mirror-like, "
            "transparent or plain objects often fail: check preview.png, and reshoot with a matte or textured object.")
    if quality == "rough":
        warnings.append(f"The best primitive fit ({best}) is only rough ({rel:.1%} of radius); treat "
                        "its dimensions as approximate.")

    fitted = {"sphere": 4 / 3 * math.pi * sph.radius ** 3, "box": float(np.prod(box.dims)),
              "cylinder": math.pi * cyl.radius ** 2 * cyl.height if cyl else None}.get(best)
    if vol is not None and fitted and quality == "good" and abs(vol - fitted) > 0.06 * fitted:
        warnings.append(
            f"Measured volume ({vol:,.0f}) and the volume of the fitted {best} ({fitted:,.0f}) differ by "
            f"{abs(vol - fitted) / fitted:.0%}. The fit is good, so prefer the fitted-shape volume: the "
            "measured one over-counts when the object's underside was never photographed.")
    fill = None if vol is None else vol / float(np.prod(obb_ext))
    res = {
        "source": {"mesh": str(mesh_path), "session": Path(mesh_path).resolve().parent.name,
                   "measured_at": datetime.now().astimezone().isoformat(timespec="seconds")},
        "units": "mm",
        "scale": {"mm_per_model_unit": S, "provided": scale_mm is not None, **(scale_info or {})},
        "plane_removal": {k: (_vec(v) if k == "plane_normal" else _r(v, 4) if isinstance(v, float) else v)
                          for k, v in iso.info.items()},
        "object": {"vertices": int(len(OV)), "triangles": int(len(OF)), "z_is_up": bool(z_up)},
        "bounding_box": {
            "axis_aligned_mm": {"x": _r(hi[0] - lo[0]), "y": _r(hi[1] - lo[1]), "z": _r(hi[2] - lo[2]),
                                "note": "Z is up" if z_up else "axes are arbitrary (mesh is not known to be Z-up)"},
            "oriented_mm": {"length": _r(obb_ext[0]), "width": _r(obb_ext[1]), "height": _r(obb_ext[2]),
                            "center": _vec(box.center, 2),
                            "note": "sorted longest to shortest along the tightest-fitting box; "
                                    "'height' is the SHORTEST side, not necessarily the vertical one",
                            "axes": [_vec(obb_R[:, k]) for k in range(3)]},
            "height_above_base_plane_mm": _r(height_above),
        },
        "volume": {"mm3": _r(vol, 1), "ml": _r(None if vol is None else vol / 1000, 2), "method": method,
                   "note": vol_note, "convex_hull_mm3": _r(hull_vol, 1)},
        "surface_area": {"mm2": _r(area, 1),
                         "note": "exposed surface only" + ("; excludes the face touching the base plane" if have_plane else "")},
        "obb_fill_ratio": _r(fill, 3),
        "obb_fill_ratio_reference": "volume / oriented-box volume: box ~1.0, cylinder ~0.785, sphere ~0.524",
        "primitive_fit": {
            "fit_points": int(samples),
            "sphere": {"center_mm": _vec(sph.center, 2), "radius_mm": _r(sph.radius), "diameter_mm": _r(2 * sph.radius),
                       "rms_residual_mm": _r(sph.rms, 3), "p95_residual_mm": _r(sph.p95, 3),
                       "rms_relative_to_radius": _r(sph.rel_rms, 4),
                       "volume_mm3": _r(4 / 3 * math.pi * sph.radius ** 3)},
            "cylinder": None if cyl is None else {
                "axis": _vec(cyl.axis), "center_mm": _vec(cyl.center, 2), "radius_mm": _r(cyl.radius),
                "diameter_mm": _r(2 * cyl.radius), "height_mm": _r(cyl.height),
                "rms_residual_mm": _r(cyl.rms, 3), "p95_residual_mm": _r(cyl.p95, 3),
                "rms_relative_to_radius": _r(cyl.rel_rms, 4), "side_surface_fraction": _r(cyl.side_fraction, 3),
                "volume_mm3": _r(math.pi * cyl.radius ** 2 * cyl.height),
                **({"axis_tilt_from_vertical_deg": _r(math.degrees(math.acos(min(1, abs(cyl.axis[2])))), 1)} if z_up else {})},
            "box": {"center_mm": _vec(box.center, 2), "dimensions_mm": _vec(box.dims),
                    "axes": [_vec(box.axes[:, k]) for k in range(3)],
                    "rms_residual_mm": _r(box.rms, 3), "p95_residual_mm": _r(box.p95, 3),
                    "rms_relative_to_half_smallest_side": _r(box.rel_rms, 4),
                    "volume_mm3": _r(float(np.prod(box.dims)))},
            "likely_shape": likely, "fit_quality": quality,
            "how_to_read": f"rms_relative (to the radius, or half the smallest side for a box) < {GOOD_REL_RMS} = good fit, "
                           f"< {ROUGH_REL_RMS} = rough, otherwise the primitive doesn't describe the object.",
        },
        "warnings": warnings,
    }
    res["summary_text"] = format_summary(res)
    return (res, OV, OF) if return_object else res
