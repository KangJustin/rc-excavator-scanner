"""Mesh -> CAD.

    python -m cad reconstructions/<session>                  # uses the scale saved by `measure`
    python -m cad reconstructions/<session> --scale-mm 79.6
    python -m cad reconstructions/<session> --shape solid    # exact non-parametric copy of the scan

Reads mesh.stl, isolates the object from the table, recognises sphere / cylinder / box, and builds a
parametric FreeCAD model (editable dimensions) plus STEP and STL. Irregular objects become a solid
built from the scan's own surface (exact, but not parametric). Reports how far the model is from the scan.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from measure import MeasureError, analyze
from reconstruct.paths import resolve_latest
from measure.__main__ import _mesh_is_z_up
from measure.scale import scale_from_known

from .appearance import vertex_colors_for
from .build import build
from .plan import SHAPES


def _saved_scale(folder: Path):
    """Scale that `python -m measure --scale-mm` recorded, if any."""
    try:
        sc = json.loads((folder / "measurements.json").read_text())["scale"]
        return sc["mm_per_model_unit"] if sc["provided"] else None
    except (OSError, KeyError, ValueError):
        return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m cad", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="reconstruction folder (contains mesh.stl) or a mesh file")
    p.add_argument("--scale-mm", type=float, default=None, metavar="MM_PER_UNIT",
                   help="millimetres per model unit (default: the value saved in measurements.json)")
    p.add_argument("--known-diameter-mm", type=float, default=None,
                   help="scale from the object's real diameter (cylinder or sphere)")
    p.add_argument("--allow-mismatch", action="store_true",
                   help="continue even if the known sizes disagree with the scan by more than 25%% (normally a hard stop)")
    p.add_argument("--known-height-mm", type=float, default=None,
                   help="real height (cylinder). With --known-diameter-mm it is an accuracy CHECK, not used for scale")
    p.add_argument("--use-known-sizes", action="store_true",
                   help="build the ideal cylinder/sphere from --known-diameter-mm/--known-height-mm instead of the scan's "
                        "(possibly lumpy) fit; the scan's deviation from those sizes is still reported")
    p.add_argument("--color", default=None, metavar="#RRGGBB",
                   help="set the model colour yourself (use when the photos were too dark or lit badly)")
    p.add_argument("--unscaled", action="store_true", help="proceed without a scale (sizes will NOT be real mm)")
    p.add_argument("--shape", choices=("auto",) + SHAPES, default="auto", help="override the detected shape")
    p.add_argument("--out", type=Path, default=None, help="output folder (default: <reconstruction>/cad)")
    p.add_argument("--name", default=None, help="base name for the files (default: the session folder name)")
    p.add_argument("--no-plane-removal", action="store_true")
    args = p.parse_args(argv)

    try:
        args.path = resolve_latest(args.path, "reconstructions", ("mesh.stl",))  # 'latest' = newest reconstruction
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    folder = args.path if args.path.is_dir() else args.path.parent
    mesh = args.path / "mesh.stl" if args.path.is_dir() else args.path
    name = args.name or (folder.name if folder.name else "model")
    out = args.out or folder / "cad"
    try:
        z_up, info = _mesh_is_z_up(folder), None
        scale = args.scale_mm
        if args.known_diameter_mm or args.known_height_mm:
            if scale is not None and not args.use_known_sizes:
                raise MeasureError("Use either --scale-mm or --known-diameter-mm/--known-height-mm, not both "
                                   "(unless you add --use-known-sizes: then --scale-mm sets the scale and the known "
                                   "sizes only set the model's dimensions).")
        if (args.known_diameter_mm or args.known_height_mm) and scale is None:
            scale, info = scale_from_known(analyze(mesh, None, not args.no_plane_removal, z_up=z_up),
                                           args.known_diameter_mm, args.known_height_mm, args.allow_mismatch)
            print(f"Scale from {info['source']}: {scale:.4g} mm per model unit (fit {info['fit_quality']})")
            for k, c in info.get("independent_check", {}).items():
                print(f"  independent check, {k}: measured {c['measured_mm']:.1f} mm vs {c['expected_mm']:g} mm "
                      f"expected ({c['error_percent']:+.1f}%)")
        if scale is None:
            scale = _saved_scale(folder)
        if scale is None and not args.unscaled:
            raise MeasureError(
                "No scale: photogrammetry has arbitrary size, so the CAD model would not be in real mm.",
                hint="Pass --scale-mm (millimetres per model unit: real size / the scan's size), or run "
                     "`python -m measure <folder> --scale-mm N` once to save it. --unscaled overrides.")
        res, OV, OF = analyze(mesh, scale, not args.no_plane_removal, z_up=z_up, return_object=True, scale_info=info)
        colors = vertex_colors_for(OV, scale or 1.0, mesh.with_suffix(".ply"))  # the reconstruction's coloured mesh
        color_override = None
        if args.color:
            h = args.color.lstrip("#")
            if len(h) != 6 or any(c not in "0123456789abcdefABCDEF" for c in h):
                raise MeasureError(f"--color must look like #c8102e, got '{args.color}'.")
            color_override = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        known = None
        if args.use_known_sizes:
            if not (args.known_diameter_mm or args.known_height_mm):
                raise MeasureError("--use-known-sizes needs --known-diameter-mm and/or --known-height-mm.")
            known = {"diameter": args.known_diameter_mm, "height": args.known_height_mm}
        report = build(res, OV, OF, out, name, str(mesh), scale is not None,
                       None if args.shape == "auto" else args.shape, colors, known, color_override)
    except (MeasureError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(f"=== CAD: {name} ===")
    print(f"Scale     : " + (f"1 model unit = {scale:g} mm" if scale is not None else "NOT SCALED (sizes are not mm)"))
    print(f"Detected  : {report['fit']}")
    kind = "parametric (edit dimensions in the Params spreadsheet)" if report["parametric"] else \
        "NOT parametric: exact solid built from the scan's surface"
    print(f"Built     : {report['shape']}, {kind}")
    for k, v in report["params_mm"].items():
        print(f"            {k.replace('_MM', '').title():9s}= {v:.2f} mm")
    if report["params_source"] != "measured from the scan":
        print(f"            (dimensions: {report['params_source']})")
    ap = report.get("appearance")
    if ap:
        pal = ", ".join(f"{c['hex']} {c['share']:.0%}" for c in ap["palette"] if "label" not in c)
        how = f" ({ap['palette'][0]['label']})" if ap["palette"][0].get("label") else ""
        print(f"Colour    : {ap['dominant_hex']}{how}; shades in the scan: {pal}")
    else:
        print("Colour    : unknown (no coloured mesh.ply next to mesh.stl), so the model is left uncoloured")
    if report.get("built"):
        b, d = report["build"], report["deviation_from_scan"]
        print(f"Volume    : {b['volume_mm3']:,.0f} mm3 ({b['volume_mm3'] / 1000:.1f} mL);  valid solid: {b['valid']}")
        print(f"Fidelity  : scan surface vs model: rms {d['rms_mm']:.2f} mm ({d['rms_fraction_of_size']:.1%} of size), "
              f"p95 {d['p95_mm']:.2f} mm, max {d['max_mm']:.2f} mm  -> {d['grade']}")
        print("Files     :")
        for k, v in report["files"].items():
            print(f"            {k:11s} {v}")
    for w in report["warnings"]:
        print(f"WARNING: {w}")
    print(f"\nReport: {out / 'cad_report.json'}")
    return 0 if report.get("built") or "macro" in report["files"] else 1


if __name__ == "__main__":
    sys.exit(main())
