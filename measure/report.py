"""Plain-text summary block: readable by a person and pasteable into an AI prompt."""
from __future__ import annotations


def _mm(x, nd=None):
    """Adaptive number format: keeps precision for tiny raw model units (0.5) and big mm values (417,990)."""
    if x is None:
        return "n/a"
    a = abs(x)
    if a == 0:
        return "0"
    if a >= 1000:
        return f"{x:,.0f}"
    if a >= 10:
        return f"{x:.1f}"
    if a >= 1:
        return f"{x:.2f}"
    return f"{x:.4g}"


def format_summary(r: dict) -> str:
    L = []
    sc, pr = r["scale"], r["plane_removal"]
    u = "mm" if sc["provided"] else "units"  # raw model units when unscaled: don't call them mm
    L.append(f"=== MEASUREMENTS: {r['source']['session']} ===")
    if sc["provided"]:
        L.append(f"Scale        : 1 model unit = {sc['mm_per_model_unit']:g} mm (user-provided)")
    else:
        L.append("Scale        : NOT SCALED. Values below are in raw model units, not real mm")
    if pr.get("removed"):
        tilt = f", tilt {pr['plane_tilt_deg']:.1f} deg from horizontal" if "plane_tilt_deg" in pr else ""
        L.append(f"Base plane   : removed ({pr['plane_inlier_fraction']:.0%} of vertices{tilt}); "
                 f"object isolated from {pr['clusters']} piece(s)")
    else:
        L.append(f"Base plane   : not removed ({pr.get('reason')})")
    o = r["object"]
    L.append(f"Object       : {o['vertices']:,} vertices, {o['triangles']:,} triangles")

    bb = r["bounding_box"]
    a, ob = bb["axis_aligned_mm"], bb["oriented_mm"]
    L += ["", f"Bounding box ({u})",
          f"  axis-aligned : X {_mm(a['x'])} x Y {_mm(a['y'])} x Z {_mm(a['z'])}   ({a['note']})",
          f"  oriented     : length {_mm(ob['length'])} x width {_mm(ob['width'])} x height {_mm(ob['height'])}"
          "   (longest to shortest)"]
    if bb["height_above_base_plane_mm"] is not None:
        L.append(f"  height above base plane : {_mm(bb['height_above_base_plane_mm'])} {u}")

    v, sa = r["volume"], r["surface_area"]
    if v["mm3"] is not None:
        L += ["", f"Volume       : {_mm(v['mm3'])} {u}3" + (f" ({_mm(v['ml'])} mL)" if sc["provided"] else "") + f"  [{v['method']}]"]
    else:
        L += ["", "Volume       : unreliable (open mesh, no base plane)"]
    if v["convex_hull_mm3"] is not None:
        L.append(f"Convex hull  : {_mm(v['convex_hull_mm3'])} {u}3 (upper bound)")
    L.append(f"Surface area : {_mm(sa['mm2'])} {u}2 ({sa['note']})")
    if r["obb_fill_ratio"] is not None:
        L.append(f"Box fill     : {r['obb_fill_ratio']:.2f} of oriented box (box ~1.0, cylinder ~0.79, sphere ~0.52)")

    pf = r["primitive_fit"]
    s, c = pf["sphere"], pf["cylinder"]
    L += ["", "Shape fits (rms = typical distance of the surface from the fitted shape)"]
    if c:
        L.append(f"  cylinder : diameter {_mm(c['diameter_mm'])} x height {_mm(c['height_mm'])} {u}, "
                 f"rms {_mm(c['rms_residual_mm'])} {u} ({c['rms_relative_to_radius']:.1%} of radius)")
        L.append(f"             axis ({', '.join(f'{t:+.2f}' for t in c['axis'])})")
    else:
        L.append("  cylinder : could not be fitted")
    L.append(f"  sphere   : diameter {_mm(s['diameter_mm'])} {u}, rms {_mm(s['rms_residual_mm'])} {u} "
             f"({s['rms_relative_to_radius']:.1%} of radius)")
    b = pf["box"]
    L.append(f"  box      : {' x '.join(_mm(x) for x in b['dimensions_mm'])} {u}, rms {_mm(b['rms_residual_mm'])} {u} "
             f"({b['rms_relative_to_half_smallest_side']:.1%} of half the smallest side)")
    L.append(f"  => likely shape: {pf['likely_shape']}  (fit quality: {pf['fit_quality']})")
    best = pf["likely_shape"].split()[0]
    if pf["fit_quality"] == "good" and pf.get(best) and pf[best].get("volume_mm3") is not None:
        L.append(f"  fitted-{best} volume: {_mm(pf[best]['volume_mm3'])} {u}3 (use this for a good fit)")

    if r["warnings"]:
        L += ["", "Warnings"] + [f"  - {w}" for w in r["warnings"]]
    return "\n".join(L)
