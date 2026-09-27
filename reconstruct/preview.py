"""Quick-look renders of the mesh, so results can be sanity-checked without a viewer.

A small software rasteriser (OpenCV, painter's algorithm) instead of Open3D's offscreen
renderer: it needs no GPU/OpenGL context, works headless, and can't take the process down.
Views are orthographic, Z-up (the mesh stage aligns to Z-up when it knows the orbit plane).
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# (label, azimuth deg, elevation deg)
VIEWS = [("front", 0, 20), ("right", 90, 20), ("back", 180, 20), ("left", 270, 20),
         ("top", 0, 80), ("underside", 0, -50)]
MAX_TRIANGLES = 60000  # decimate above this so preview stays quick
SS = 2  # supersampling factor for anti-aliasing


def render_view(V: np.ndarray, F: np.ndarray, face_rgb: np.ndarray, face_n: np.ndarray,
                az: float, el: float, size: int, radius: float, center: np.ndarray) -> np.ndarray:
    """One orthographic view. Returns a size x size BGR uint8 image."""
    a, e = np.radians(az), np.radians(el)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])  # towards camera
    z = np.array([0.0, 0.0, 1.0])
    right = np.cross(-d, z)
    right /= np.linalg.norm(right)
    up = np.cross(right, -d)

    P = V - center
    scale = 0.45 * size * SS / radius
    x = size * SS / 2 + (P @ right) * scale
    y = size * SS / 2 - (P @ up) * scale
    depth = P @ d  # larger = nearer the camera

    light = d + 0.6 * z
    light /= np.linalg.norm(light)
    shade = 0.30 + 0.70 * np.abs(face_n @ light)  # two-sided so flipped normals don't go black

    tri_depth = depth[F].mean(axis=1)
    order = np.argsort(tri_depth)  # far to near
    img = np.full((size * SS, size * SS, 3), 245, np.uint8)
    pts2d = np.stack([x, y], axis=1)
    colors = np.clip(face_rgb * shade[:, None] * 255, 0, 255).astype(np.uint8)
    for i in order:
        poly = pts2d[F[i]].astype(np.int32)
        c = colors[i]
        cv2.fillConvexPoly(img, poly, (int(c[2]), int(c[1]), int(c[0])))  # RGB -> BGR
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def render_preview(mesh_path: Path, out_png: Path, size: int = 420) -> Path:
    """Render the mesh from six angles into one contact sheet PNG."""
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"{mesh_path} has no triangles")
    if len(mesh.triangles) > MAX_TRIANGLES:
        mesh = mesh.simplify_quadric_decimation(MAX_TRIANGLES)
    mesh.compute_triangle_normals()

    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    n = np.asarray(mesh.triangle_normals)
    if mesh.has_vertex_colors():
        face_rgb = np.asarray(mesh.vertex_colors)[F].mean(axis=1)
    else:
        face_rgb = np.full((len(F), 3), 0.75)

    lo, hi = V.min(axis=0), V.max(axis=0)
    center = (lo + hi) / 2
    radius = float(np.linalg.norm(V - center, axis=1).max()) or 1.0

    tiles = []
    for label, az, el in VIEWS:
        tile = render_view(V, F, face_rgb, n, az, el, size, radius, center)
        cv2.putText(tile, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2, cv2.LINE_AA)
        cv2.rectangle(tile, (0, 0), (size - 1, size - 1), (200, 200, 200), 1)
        tiles.append(tile)
    sheet = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_png), sheet):
        raise OSError(f"could not write {out_png}")
    return out_png
