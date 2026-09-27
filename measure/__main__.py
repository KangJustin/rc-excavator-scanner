"""Real-world dimensions from a reconstructed mesh, for feeding to an AI / CAD.

    python -m measure reconstructions/<session-name> --scale-mm 65.4

Reads <folder>/mesh.stl, prints a summary block, and writes <folder>/measurements.json.
--scale-mm is millimetres per model unit: photogrammetry has arbitrary scale, so measure
one real dimension of the object (say it's 122 mm tall while the mesh says 1.87) and pass 122/1.87.
(If you already scaled the mesh with `reconstruct --scale`, don't scale it twice: pass 1.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reconstruct.paths import resolve_latest

from .analyze import MeasureError, analyze
from .scale import scale_from_known


def _mesh_is_z_up(folder: Path) -> bool:
    """reconstruct records whether it rotated the mesh Z-up in report.json."""
    try:
        return bool(json.loads((folder / "report.json").read_text())["mesh"]["aligned"])
    except (OSError, KeyError, TypeError, ValueError):
        return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m measure", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="reconstruction folder (contains mesh.stl) or a mesh file")
    p.add_argument("--scale-mm", type=float, default=None, metavar="MM_PER_UNIT",
                   help="millimetres per model unit (default: unscaled, reported in model units)")
    p.add_argument("--known-diameter-mm", type=float, default=None,
                   help="scale from the object's real diameter (cylinder or sphere), instead of --scale-mm")
    p.add_argument("--allow-mismatch", action="store_true",
                   help="continue even if the known sizes disagree with the scan by more than 25%% (normally a hard stop)")
    p.add_argument("--known-height-mm", type=float, default=None,
                   help="real height (cylinder). With --known-diameter-mm it is used as an accuracy CHECK")
    p.add_argument("--no-plane-removal", action="store_true", help="measure the whole mesh")
    p.add_argument("--plane-tol", type=float, default=0.008,
                   help="plane thickness as a fraction of the mesh's bounding-box diagonal (default 0.008)")
    p.add_argument("--samples", type=int, default=20000, help="surface points used for shape fitting")
    p.add_argument("--json-only", action="store_true", help="print the JSON instead of the text block")
    args = p.parse_args(argv)

    try:
        args.path = resolve_latest(args.path, "reconstructions", ("mesh.stl",))  # 'latest' = newest reconstruction
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    folder = args.path if args.path.is_dir() else args.path.parent
    mesh = args.path / "mesh.stl" if args.path.is_dir() else args.path
    try:
        if args.path.is_dir() and not mesh.exists():
            hint = ("This looks like a folder of photos. Measure a reconstruction instead: run "
                    "`python -m reconstruct <this folder>`, then `python -m measure reconstructions/<name>`."
                    if list(args.path.glob("*.jpg")) else
                    "Run `python -m reconstruct captures/<session>` first; it writes mesh.stl.")
            raise MeasureError(f"No mesh.stl in {args.path}", hint=hint)
        z_up, scale, info = _mesh_is_z_up(folder), args.scale_mm, None
        if args.known_diameter_mm or args.known_height_mm:
            if scale is not None:
                raise MeasureError("Use either --scale-mm or --known-diameter-mm/--known-height-mm, not both.")
            scale, info = scale_from_known(analyze(mesh, None, not args.no_plane_removal, args.plane_tol, args.samples,
                                                   z_up=z_up), args.known_diameter_mm, args.known_height_mm, args.allow_mismatch)
            print(f"Scale from {info['source']}: {scale:.4g} mm per model unit (fit {info['fit_quality']})")
            for k, c in info.get("independent_check", {}).items():
                print(f"  independent check, {k}: measured {c['measured_mm']:.1f} mm vs {c['expected_mm']:g} mm "
                      f"expected ({c['error_percent']:+.1f}%)")
        res = analyze(mesh, scale, not args.no_plane_removal, args.plane_tol, args.samples, z_up=z_up,
                      scale_info=info)
    except MeasureError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    out = folder / "measurements.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2) if args.json_only else res["summary_text"])
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
