"""Write the macro, run it in headless FreeCAD, and verify the result against the scan."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .appearance import dominant_colors
from .guistate import add_gui_state
from .plan import make_plan
from .templates import macro_source
from .verify import deviation

CANDIDATES = ["/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd", "freecadcmd", "FreeCADCmd"]


def find_freecadcmd() -> str | None:
    for c in CANDIDATES:
        p = shutil.which(c) or (c if Path(c).exists() else None)
        if p:
            return p
    return None


def _set_var(src: str, name: str, value) -> str:
    out, n = re.subn(rf"^{name} = .*$", f"{name} = {value!r}", src, flags=re.M)
    assert n == 1, name
    return out


def _mesh_volume(V, F) -> float:
    tri = V[F] - V.mean(axis=0)
    return abs(float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0))


def _closed(m) -> bool:
    return bool(m.is_edge_manifold(False) and m.is_vertex_manifold())


def _consistently_oriented(F) -> bool:
    """A closed, consistently wound mesh uses every directed edge exactly once, and its reverse once.
    (Open3D's is_orientable() missed a mesh with 83 flipped edges, so we check it ourselves.)"""
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    directed = {}
    for a, b in map(tuple, e):
        directed[(a, b)] = directed.get((a, b), 0) + 1
    return all(c == 1 and (b, a) in directed for (a, b), c in directed.items())


def _boundary_edges(F):
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    present = set(map(tuple, e))
    return [(a, b) for a, b in map(tuple, e) if (b, a) not in present]


def _boundary_loops(F):
    """Closed loops of boundary edges, each as an ordered list of vertex ids (in the mesh's own edge direction)."""
    nxt = {}
    for a, b in _boundary_edges(F):
        nxt.setdefault(a, []).append(b)
    loops, seen = [], set()
    for start in list(nxt):
        if start in seen:
            continue
        loop, cur = [start], nxt[start][0]
        seen.add(start)
        while cur != start and cur not in seen and cur in nxt:
            loop.append(cur)
            seen.add(cur)
            cur = nxt[cur][0]
        if cur == start and len(loop) >= 3:
            loops.append(loop)
    return loops


def _ear_clip(P, poly):
    """Triangulate a simple polygon (2D points P, vertex ids in `poly`, in order). Triangles keep the polygon's
    order, so they are wound consistently with it. Returns None if it can't (self-intersecting input)."""
    idx = list(poly)
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    area = sum(P[idx[i]][0] * P[idx[(i + 1) % len(idx)]][1] - P[idx[(i + 1) % len(idx)]][0] * P[idx[i]][1]
               for i in range(len(idx)))
    sgn = 1.0 if area > 0 else -1.0
    tris = []
    while len(idx) > 3:
        for i in range(len(idx)):
            a, b, c = idx[i - 1], idx[i], idx[(i + 1) % len(idx)]
            if sgn * cross(P[a], P[b], P[c]) <= 1e-12:  # reflex or collinear corner: not an ear
                continue
            inside = False
            for q in idx:
                if q in (a, b, c):
                    continue
                d1, d2, d3 = cross(P[a], P[b], P[q]), cross(P[b], P[c], P[q]), cross(P[c], P[a], P[q])
                if sgn * d1 >= -1e-12 and sgn * d2 >= -1e-12 and sgn * d3 >= -1e-12:
                    inside = True
                    break
            if not inside:
                tris.append([a, b, c])
                idx.pop(i)
                break
        else:
            return None
    tris.append(idx)
    return tris


def _cap_planar_loops(V, F, flatness=0.05):
    """Cap every boundary loop that is close to flat (checked against its own best-fit plane) by ear clipping.
    Deterministic, and it handles concave outlines. Curved holes are left for the caller."""
    caps = []
    for loop in _boundary_loops(F):
        pts = V[loop]
        centre = pts.mean(axis=0)
        _, sv, Vt = np.linalg.svd(pts - centre)
        normal = Vt[2]
        extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) or 1.0
        if float(np.abs((pts - centre) @ normal).max()) > flatness * extent:
            continue
        P2 = np.zeros((len(V), 2))
        P2[loop] = np.c_[(pts - centre) @ Vt[0], (pts - centre) @ Vt[1]]
        t = _ear_clip(P2, loop[::-1])  # the cap must run opposite to the loop's edges
        if t:
            caps += t
    return np.vstack([F, np.array(caps)]) if caps else F


def close_object(V, F, plane: dict | None, target=6000):
    """Turn the isolated (open-bottomed) object into a closed, consistently oriented, outward-facing mesh.

    1. If the object was cut off a base plane, extend its cut edge down to that plane (measure removes a
       thin slab of the base when it cuts the table away; without this the solid would float above the floor).
    2. Close every remaining hole (the flat bottom, plus any small gaps).
    3. Re-orient all triangles consistently and outward.
    4. Thin it out for FreeCAD, keeping the thinned mesh only if it is still closed and consistent.
    Returns (V, F, ok)."""
    import open3d as o3d
    V, F = np.asarray(V, float), np.asarray(F, int)
    if plane and plane.get("removed"):
        n, d = np.array(plane["plane_normal"]), plane["plane_offset"]
        h = V @ n + d
        edges = [(a, b) for a, b in _boundary_edges(F) if h[a] < 5 * plane["tolerance"] and h[b] < 5 * plane["tolerance"]]
        if edges:
            used = sorted({v for e in edges for v in e})
            new_id = {v: len(V) + k for k, v in enumerate(used)}
            V = np.vstack([V, V[used] - (h[used][:, None] * n)])  # the same vertices, dropped onto the plane
            skirt = []
            for a, b in edges:  # wall between the cut edge and its projection, wound opposite to the existing edge
                skirt += [[b, a, new_id[a]], [b, new_id[a], new_id[b]]]
            F = np.vstack([F, np.array(skirt)])

    F = _cap_planar_loops(V, F)  # flat holes (the base, small planar gaps): exact and deterministic
    extent = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    if _boundary_edges(F):  # anything still open (small gaps): let Open3D fill it, but never trust it blindly
        filled = o3d.t.geometry.TriangleMesh.from_legacy(m).fill_holes(hole_size=1e9).to_legacy()
        fv = np.asarray(filled.vertices)
        if np.isfinite(fv).all() and np.abs(fv - V.mean(axis=0)).max() < 2 * extent:
            m = filled
    m.remove_duplicated_vertices(); m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()

    def finish(mesh):
        mesh.orient_triangles()
        Vv, Ff = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        t = Vv[Ff] - Vv.mean(axis=0)
        if np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() < 0:  # inside-out: flip
            Ff = Ff[:, ::-1].copy()
        return Vv, Ff, _closed(mesh) and _consistently_oriented(Ff)

    Vc, Fc, ok = finish(m)
    if len(Fc) > target:
        thin = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Vc), o3d.utility.Vector3iVector(Fc)).simplify_quadric_decimation(target)
        thin.remove_degenerate_triangles(); thin.remove_unreferenced_vertices()
        Vt, Ft, ok_t = finish(thin)
        if ok_t or not ok:
            Vc, Fc, ok = Vt, Ft, ok_t
    return Vc, Fc, ok


def build(res: dict, OV: np.ndarray, OF: np.ndarray, out_dir: Path, name: str, source: str,
          scale_confirmed: bool, force: str | None = None, colors: np.ndarray | None = None,
          known: dict | None = None, color_override: tuple | None = None) -> dict:
    """Returns a report dict (also written to cad_report.json). `colors` = per-vertex RGB (0..1) of the scan
    object, if known; it gives the model its colour and produces the appearance picture."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if known and force is None and known.get("diameter") and known.get("height") \
            and res["primitive_fit"]["likely_shape"].startswith("neither"):
        force = "cylinder"  # a diameter AND a height describe a cylinder, and the scan is too lumpy to say otherwise
        note_inferred = True
    else:
        note_inferred = False
    plan = make_plan(res, OV, force)
    shape = plan["shape"]
    params_source = "measured from the scan"
    if known:  # build the ideal shape from sizes the user knows, not from the (possibly lumpy) scan's fit
        if shape == "sphere" and known.get("diameter"):
            plan["params"]["DIAMETER_MM"] = float(known["diameter"])
        elif shape == "cylinder":
            for key, param in (("diameter", "DIAMETER_MM"), ("height", "HEIGHT_MM")):
                if known.get(key):
                    plan["params"][param] = float(known[key])
        else:
            raise ValueError(f"--use-known-sizes works for a sphere or cylinder, not '{shape}'.")
        params_source = "the real sizes you gave (--use-known-sizes), NOT the scan; the scan's deviation from them is reported below"
    pose = np.array(plan["pose"])
    report = {"shape": shape, "detected": plan["detected"], "fit": plan["fit_note"], "params_mm": plan["params"],
              "pose_cad_to_scan": plan["pose"], "parametric": shape != "solid", "warnings": [], "files": {},
              "params_source": params_source}

    if note_inferred:
        report["warnings"].append("The scan was too lumpy to classify, so a cylinder was assumed because you gave both a "
                                  "diameter and a height.")
    mesh_path = expected = None
    if shape == "solid":
        import open3d as o3d
        from .verify import to_cad_frame
        cV, cF, ok = close_object(OV, OF, res["plane_removal"])
        if not ok:
            report["warnings"].append("Could not turn the scan into a fully closed, consistently oriented mesh; the "
                                      "solid conversion may fail or be unreliable.")
        plan = make_plan(res, cV, "solid")  # the pose comes from the CLOSED mesh, so it stands on the plane
        pose = np.array(plan["pose"])
        report["pose_cad_to_scan"] = plan["pose"]
        cV = to_cad_frame(cV, pose)  # store the solid in the CAD frame (standing on Z=0)
        mesh_path = out_dir / f"{name}_closed_mesh.stl"
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(cV), o3d.utility.Vector3iVector(cF))
        m.compute_triangle_normals()
        o3d.io.write_triangle_mesh(str(mesh_path), m)
        expected = _mesh_volume(cV, cF)
        report["files"]["closed_mesh"] = str(mesh_path)

    palette = dominant_colors(colors) if colors is not None and len(colors) else None
    color = palette[0]["rgb"] if palette else None
    if color_override is not None:
        color = tuple(int(v) for v in color_override)
        palette = [{"rgb": color, "hex": "#%02x%02x%02x" % color, "share": 1.0, "label": "set by you (--color)"}] + (palette or [])
    if color is not None and 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2] < 46 and color_override is None:
        report["warnings"].append(
            f"The scan's colours are very dark (dominant #{color[0]:02x}{color[1]:02x}{color[2]:02x}). The photos were "
            "probably underexposed (add light and reshoot), or use --color '#RRGGBB' to set the true colour yourself.")
    if palette:
        report["appearance"] = {"dominant_rgb": list(color), "dominant_hex": palette[0]["hex"],
                                "palette": palette,
                                "note": "Colours as seen in the photos (lighting is baked in); highlights and shadows trimmed."}
    macro = macro_source(plan, name, source, scale_confirmed, mesh_path, expected, color)
    macro_path = out_dir / f"{name}.FCMacro"
    macro_path.write_text(macro)
    report["files"]["macro"] = str(macro_path)

    fc = find_freecadcmd()
    if fc is None:
        report["warnings"].append("FreeCAD (freecadcmd) not found: wrote the macro only. Open it in FreeCAD "
                                  "(Macro > Macros... > Execute) to build the model.")
        report["built"] = False
        return _finish(report, out_dir)

    paths = {"EXPORT_STL_PATH": out_dir / f"{name}.stl", "EXPORT_STEP_PATH": out_dir / f"{name}.step",
             "SAVE_FCSTD_PATH": out_dir / f"{name}.FCStd", "REPORT_JSON_PATH": out_dir / f"{name}_build.json"}
    run_src = macro
    for k, v in paths.items():
        run_src = _set_var(run_src, k, str(v))
    runner = out_dir / f".{name}_run.py"
    runner.write_text(run_src)
    proc = subprocess.run([fc, str(runner)], capture_output=True, text=True, timeout=600)
    runner.unlink(missing_ok=True)
    if proc.returncode != 0 or not paths["REPORT_JSON_PATH"].exists():
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-8:])
        report["built"] = False
        report["warnings"].append(f"FreeCAD failed to build the model:\n{tail}")
        return _finish(report, out_dir)

    build_info = json.loads(paths["REPORT_JSON_PATH"].read_text())
    paths["REPORT_JSON_PATH"].unlink()
    add_gui_state(paths["SAVE_FCSTD_PATH"], build_info["bbox_mm"], color)  # so it opens showing the model, not an empty view
    report.update(built=True, build=build_info)
    report["files"].update(fcstd=str(paths["SAVE_FCSTD_PATH"]), step=str(paths["EXPORT_STEP_PATH"]),
                           stl=str(paths["EXPORT_STL_PATH"]))
    report["deviation_from_scan"] = deviation(OV, OF, pose, paths["EXPORT_STL_PATH"])
    if report["deviation_from_scan"]["grade"] in ("rough", "poor"):
        report["warnings"].append(
            f"The {shape} model deviates noticeably from the scan (rms {report['deviation_from_scan']['rms_mm']:.2f} mm, "
            f"{report['deviation_from_scan']['rms_fraction_of_size']:.1%} of size). "
            + ("Try `--shape solid` for an exact (non-parametric) copy." if shape != "solid" else ""))
    try:
        report["files"]["preview"] = _preview(paths["EXPORT_STL_PATH"], out_dir / f"{name}_preview.png", color)
        if colors is not None:
            report["files"].update(_appearance_files(OV, OF, colors, paths["EXPORT_STL_PATH"], color, out_dir, name))
    except Exception as e:  # never lose the model over a picture
        report["warnings"].append(f"Preview not rendered: {e}")
    return _finish(report, out_dir)


def _preview(stl: Path, png: Path, color=None) -> str:
    import open3d as o3d
    from reconstruct.preview import render_preview
    m = o3d.io.read_triangle_mesh(str(stl))
    rgb = np.array(color if color else (217, 115, 38), float) / 255.0  # orange when the colour is unknown
    m.vertex_colors = o3d.utility.Vector3dVector(np.tile(rgb, (len(m.vertices), 1)))
    tmp = png.with_suffix(".ply")
    o3d.io.write_triangle_mesh(str(tmp), m)
    render_preview(tmp, png)
    tmp.unlink(missing_ok=True)
    return str(png)


def _appearance_files(OV, OF, colors, cad_stl: Path, color, out_dir: Path, name: str) -> dict:
    """What it actually looks like: the scan object in its real (photographed) colours as a 3D file, and a
    picture comparing it with the CAD model."""
    import cv2
    import open3d as o3d
    from reconstruct.preview import render_preview
    scan = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(OV), o3d.utility.Vector3iVector(OF))
    scan.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    scan.compute_vertex_normals()
    ply = out_dir / f"{name}_scan_colored.ply"
    o3d.io.write_triangle_mesh(str(ply), scan)
    tmp = out_dir / ".scan_sheet.png"
    render_preview(ply, tmp)
    cad_png = Path(_preview(cad_stl, out_dir / ".cad_sheet.png", color))
    a, b = cv2.imread(str(tmp)), cv2.imread(str(cad_png))
    def titled(img, text):
        bar = np.full((44, img.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, text, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA)
        return np.vstack([bar, img])
    sheet = np.vstack([titled(a, "SCAN, as photographed (true colours)"), titled(b, "CAD MODEL (dominant colour)")])
    out = out_dir / f"{name}_appearance.png"
    cv2.imwrite(str(out), sheet)
    tmp.unlink(missing_ok=True)
    cad_png.unlink(missing_ok=True)
    return {"appearance": str(out), "scan_colored_ply": str(ply)}


def _finish(report: dict, out_dir: Path) -> dict:
    (out_dir / "cad_report.json").write_text(json.dumps(report, indent=2))
    return report
