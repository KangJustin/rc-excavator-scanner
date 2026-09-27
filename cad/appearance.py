"""What the object actually looks like: its colours, taken from the scan's vertex colours.

Photogrammetry bakes the lighting of the shoot into these colours (highlights on shiny surfaces, shadows
underneath), so the result is *how it looked in the photos*, not a perfect paint colour. Highlights and
shadows are trimmed before finding the dominant colour, which limits (not removes) that bias.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def vertex_colors_for(object_vertices: np.ndarray, scale_mm: float, ply_path: Path) -> np.ndarray | None:
    """Colour (0..1 RGB) for each object vertex, looked up from the reconstruction's coloured mesh.ply.
    The PLY is in the same frame as the STL but in model units, so the vertices are scaled back first."""
    import open3d as o3d
    if not Path(ply_path).exists():
        return None
    ply = o3d.io.read_triangle_mesh(str(ply_path))
    if not ply.has_vertex_colors():
        return None
    src = np.asarray(ply.vertices)
    q = np.asarray(object_vertices) / float(scale_mm)
    nns = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(src.astype(np.float64)))
    nns.knn_index()
    idx, dist = nns.knn_search(o3d.core.Tensor(q.astype(np.float64)), 1)
    idx, dist = idx.numpy().ravel(), dist.numpy().ravel()
    if np.median(dist) > 0.01 * max(1e-9, float(np.linalg.norm(src.max(axis=0) - src.min(axis=0)))):
        return None  # the PLY does not line up with the STL: don't invent colours
    return np.asarray(ply.vertex_colors)[idx]


def dominant_colors(colors: np.ndarray, k: int = 3, seed: int = 0) -> list[dict]:
    """Palette of the object as [{'rgb': (r,g,b) 0-255, 'hex', 'share'}], biggest first.

    The FIRST entry is the base colour: the largest cluster if it covers at least half the object, otherwise the
    average colour of the object (labelled as such). Drops the brightest 8% (specular highlights) and the darkest 8% (deep shadow), then clusters on CHROMATICITY
    (which colour, ignoring how bright it is). Clustering on raw RGB would split one orange ball into a
    'dark orange', an 'orange' and a 'light orange' just because of lighting. Each entry's colour is the mean
    of its members, so it keeps the object's average brightness."""
    c = np.clip(np.asarray(colors, float), 0, 1)
    lum = c @ np.array([0.2126, 0.7152, 0.0722])
    lo, hi = np.percentile(lum, [8, 92])
    keep = c[(lum >= lo) & (lum <= hi)]
    if len(keep) < k * 10:
        keep = c
    chroma = keep[:, :2] / (keep.sum(axis=1, keepdims=True) + 1e-9)  # (r, g) share of the total; b = 1 - r - g
    rng = np.random.default_rng(seed)
    centers = chroma[rng.choice(len(chroma), k, replace=False)]
    for _ in range(30):
        lab = np.argmin(((chroma[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        new = np.array([chroma[lab == j].mean(axis=0) if (lab == j).any() else centers[j] for j in range(k)])
        if np.allclose(new, centers, atol=1e-5):
            break
        centers = new
    lab = np.argmin(((chroma[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
    out = []
    for j in np.argsort(-np.bincount(lab, minlength=k)):
        if not (lab == j).any():
            continue
        rgb = tuple(int(round(v * 255)) for v in keep[lab == j].mean(axis=0))
        out.append({"rgb": rgb, "hex": "#%02x%02x%02x" % rgb, "share": float((lab == j).mean())})
    if out and out[0]["share"] < 0.5:  # no single colour covers half the object: it is shades of one colour
        rgb = tuple(int(round(v * 255)) for v in keep.mean(axis=0))  # (or a genuinely mixed object): use the average
        out.insert(0, {"rgb": rgb, "hex": "#%02x%02x%02x" % rgb, "share": 1.0, "label": "average of the object"})
    return out
