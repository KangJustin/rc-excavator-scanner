"""Folder of photos -> COLMAP sparse -> dense (COLMAP or OpenMVS) -> mesh (.ply/.stl) + preview.png

    python -m reconstruct --doctor                              # check COLMAP / OpenMVS / Open3D install first
    python -m reconstruct latest                                # the newest photo session in ./captures
    python -m reconstruct captures/2026-09-25_2130
    python -m reconstruct <images> --dense-backend openmvs      # CPU dense (no NVIDIA GPU)
    python -m reconstruct <images> --only sparse                # run one stage; workspace is the hand-off
    python -m reconstruct <images> --only mesh --remove-plane --largest-cluster
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .errors import ReconstructionError
from .meshing import MeshOptions
from . import config
from .paths import resolve_latest
from .pipeline import STAGES, ReconstructionConfig, run_reconstruction


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m reconstruct", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image_dir", type=Path, nargs="?", help="folder of photos (not needed with --doctor)")
    p.add_argument("--doctor", action="store_true",
                   help="check the install (COLMAP, OpenMVS, Open3D) and which dense backend auto would pick, then exit")
    p.add_argument("--out", type=Path,
                   help=f"workspace folder (default: ./{config.OUTPUT_ROOT}/<image folder name>)")
    p.add_argument("--only", choices=STAGES, help="run just one stage against an existing workspace")
    p.add_argument("--colmap-bin", default=config.COLMAP_BIN,
                   help="COLMAP executable (default: COLMAP_BIN in reconstruct/config.py or $COLMAP_BIN)")
    p.add_argument("--dense-backend", choices=("auto", "colmap", "openmvs"), default=config.DENSE_BACKEND,
                   help="colmap = CUDA GPU; openmvs = CPU; auto = colmap if possible, else openmvs")
    p.add_argument("--openmvs-bin-dir", default=config.OPENMVS_BIN_DIR,
                   help="folder holding InterfaceCOLMAP and DensifyPointCloud (default: search PATH)")
    p.add_argument("--allow-partial", action="store_true",
                   help="continue even if the camera path is a straight line / covers < 120 degrees")
    p.add_argument("--no-preview", action="store_true", help="skip rendering preview.png")
    p.add_argument("--cpu-only", action="store_true", help="force CPU SIFT extraction/matching")
    p.add_argument("--camera-model", default="SIMPLE_RADIAL",
                   help="COLMAP camera model (default SIMPLE_RADIAL, fine for a webcam)")
    p.add_argument("-v", "--verbose", action="store_true", help="echo COLMAP output")

    g = p.add_argument_group("meshing")
    g.add_argument("--poisson-depth", type=int, default=9)
    g.add_argument("--poisson-threads", type=int, default=1,
                   help="threads for Poisson (default 1: Open3D's multithreaded Poisson crashes "
                        "intermittently on macOS arm64; -1 = all cores)")
    g.add_argument("--trim", type=float, default=0.02,
                   help="drop this fraction of least-supported mesh vertices (default 0.02; 0 keeps all)")
    g.add_argument("--voxel-size", type=float, help="downsample the cloud first (model units)")
    g.add_argument("--crop", type=float, nargs=6,
                   metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
                   help="keep only points inside this box (model units; see cleaned/sparse cloud)")
    g.add_argument("--remove-plane", action="store_true", help="RANSAC-remove the dominant plane (table)")
    g.add_argument("--largest-cluster", action="store_true", help="keep only the largest connected blob")
    g.add_argument("--no-align", action="store_true",
                   help="don't rotate the mesh to Z-up / sit it on z=0 (keep COLMAP's coordinate frame)")
    g.add_argument("--scale", type=float, default=1.0,
                   help="multiply the mesh by this factor (SfM scale is arbitrary)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.doctor:
        from .doctor import format_report, run_doctor
        report = run_doctor(args.colmap_bin, args.openmvs_bin_dir, args.dense_backend)
        print(format_report(report))
        return 0 if report.ready else 1
    if args.image_dir is None:
        p.error("image_dir is required (or use --doctor)")
    try:
        args.image_dir = resolve_latest(args.image_dir, "captures", ("*.jpg", "*.jpeg", "*.png"))
    except FileNotFoundError as e:
        p.error(str(e))
    cfg = ReconstructionConfig(
        image_dir=args.image_dir, out_dir=args.out, only=args.only,
        colmap_bin=args.colmap_bin, openmvs_bin_dir=args.openmvs_bin_dir,
        dense_backend=args.dense_backend, cpu_only=args.cpu_only, camera_model=args.camera_model,
        allow_partial=args.allow_partial, preview=not args.no_preview, verbose=args.verbose,
        mesh=MeshOptions(poisson_depth=args.poisson_depth, poisson_threads=args.poisson_threads, density_trim_quantile=args.trim,
                         voxel_size=args.voxel_size, crop=tuple(args.crop) if args.crop else None,
                         remove_plane=args.remove_plane, largest_cluster=args.largest_cluster,
                         scale=args.scale, align_up=not args.no_align),
    )
    try:
        r = run_reconstruction(cfg)
    except ReconstructionError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 2

    print(f"\nWorkspace: {r.workspace}")
    if r.registered_images is not None:
        print(f"Sparse: {r.registered_images}/{r.total_images} images registered, "
              f"{r.sparse_points} points  (view {r.workspace / 'sparse_points.ply'})")
    if r.coverage_deg is not None:
        print(f"        cameras cover {r.coverage_deg:.0f} of 360 degrees")
    if r.dense_points is not None:
        print(f"Dense:  {r.dense_points} points" + (f" (backend: {r.dense_backend})" if r.dense_backend else ""))
    if r.mesh:
        m = r.mesh
        print(f"Mesh:   {m.vertices} vertices, {m.triangles} triangles, "
              f"watertight={m.watertight}")
        print("        extents (model units): %.4g x %.4g x %.4g%s  <- SfM scale is arbitrary; "
              "use --scale to make this real mm" % (*m.extents, " (W x D x H, Z-up)" if m.aligned else ""))
        print(f"        {m.ply}\n        {m.stl}")
    if r.preview:
        print(f"Preview: {r.preview}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
