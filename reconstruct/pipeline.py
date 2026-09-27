"""Images -> COLMAP sparse -> dense (COLMAP or OpenMVS) -> Poisson mesh (.ply + .stl) -> preview.

Stages can be run separately (`only=`) against the same workspace folder, so e.g. the
sparse stage can run on a laptop and the dense stage on a GPU machine:

    reconstructions/<session>/
        database.db          COLMAP features + matches
        sparse/best/         the sparse model used downstream (+ sparse_points.ply for viewing)
        scene.json           camera-path analysis: up vector, object centre, coverage
        dense/               undistorted images, depth maps, fused.ply
        cleaned_points.ply, mesh.ply, mesh.stl, preview.png
        logs/*.log, report.json
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import colmap as cm
from .config import COLMAP_BIN, DENSE_BACKEND, OPENMVS_BIN_DIR, OUTPUT_ROOT
from .dense import AUTO_ORDER, BACKENDS, DenseContext
from .errors import (
    ColmapCommandError, DegenerateReconstructionError, DenseReconstructionError,
    DenseUnavailableError, InsufficientOverlapError, OpenMVSNotFoundError, PreflightError,
    ReconstructionError, SparseReconstructionError, TooFewFeaturesError,
)
from .geometry import CoverageReport, analyze_camera_coverage
from .meshing import MeshOptions, MeshResult, build_mesh
from .preflight import preflight

log = logging.getLogger(__name__)

STAGES = ("sparse", "dense", "mesh", "preview")

MIN_KEYPOINTS_WARN = 300  # per image
MIN_KEYPOINTS_MEDIAN_FATAL = 100
MIN_REGISTERED_FRACTION = 0.6
MIN_DENSE_POINTS = 1000

LINEARITY_FATAL = 0.06  # camera centres this close to collinear = "moved in a straight line"
COVERAGE_FATAL_DEG = 120.0  # cameras span less than this around the object = not enough angle
COVERAGE_WARN_DEG = 300.0
ELEVATION_WARN_DEG = 8.0

TEXTURE_HINT = (
    "Shiny, plain or transparent objects (soda cans!) give SIFT almost nothing to lock "
    "onto. Put the object on a busy, matte, non-repeating surface (newspaper, a printed "
    "pattern, a rug), add matte tape or marker dots to the object, use diffuse light "
    "without glare, and check for blur."
)


@dataclass
class ReconstructionConfig:
    image_dir: Path
    out_dir: Path | None = None  # default: ./reconstructions/<image folder name>
    only: str | None = None  # None = sparse+dense+mesh+preview, else one of STAGES
    colmap_bin: str = COLMAP_BIN
    openmvs_bin_dir: str = OPENMVS_BIN_DIR
    dense_backend: str = DENSE_BACKEND  # "auto" | "colmap" | "openmvs"
    cpu_only: bool = False  # force CPU SIFT (extraction/matching)
    camera_model: str = "SIMPLE_RADIAL"
    allow_partial: bool = False  # downgrade "not enough angular coverage / straight line" to a warning
    preview: bool = True
    verbose: bool = False
    mesh: MeshOptions = field(default_factory=MeshOptions)


@dataclass
class ReconstructionResult:
    workspace: Path
    registered_images: int | None = None
    total_images: int | None = None
    sparse_points: int | None = None
    coverage_deg: float | None = None
    dense_backend: str | None = None
    dense_points: int | None = None
    mesh: MeshResult | None = None
    preview: Path | None = None


def run_reconstruction(cfg: ReconstructionConfig) -> ReconstructionResult:
    image_dir = Path(cfg.image_dir).resolve()
    if cfg.only is not None and cfg.only not in STAGES:
        raise ReconstructionError(f"Unknown stage '{cfg.only}'.", hint=f"Choose one of: {', '.join(STAGES)}")
    stages = ("sparse", "dense", "mesh") if cfg.only is None else (cfg.only,)
    if ("sparse" in stages or "dense" in stages) and not image_dir.is_dir():
        raise PreflightError(f"Image folder not found: {image_dir}")

    ws = Path(cfg.out_dir or Path(OUTPUT_ROOT) / image_dir.name).resolve()
    if ws == image_dir or image_dir in ws.parents:
        raise PreflightError(
            f"Output folder {ws} is inside the image folder.",
            hint="COLMAP scans the image folder recursively and would ingest its own output. "
                 "Choose an --out outside it.")
    colmap = cm.Colmap(cfg.colmap_bin, ws / "logs", cfg.verbose)
    if "sparse" in stages or "dense" in stages:
        colmap.ensure_available()  # before creating anything on disk
    ws.mkdir(parents=True, exist_ok=True)
    result = ReconstructionResult(workspace=ws)

    if "sparse" in stages:
        _sparse(cfg, colmap, image_dir, ws, result)
    if "dense" in stages:
        _dense(cfg, colmap, image_dir, ws, result)
    if "mesh" in stages:
        log.info("== Poisson meshing ==")
        scene_path = ws / "scene.json"
        scene = json.loads(scene_path.read_text()) if scene_path.exists() else None
        result.mesh = build_mesh(ws / "dense" / "fused.ply", ws, cfg.mesh, scene)
    if "preview" in stages or ("mesh" in stages and cfg.preview):
        result.preview = _preview(ws)
    _write_report(ws, result)
    return result


# ---------------------------------------------------------------------------
def _sparse(cfg, colmap: cm.Colmap, image_dir: Path, ws: Path, result: ReconstructionResult):
    log.info("== Sparse reconstruction ==")
    images = preflight(image_dir)
    result.total_images = len(images)
    db, sparse = ws / "database.db", ws / "sparse"
    for p in (db, sparse, ws / "sparse_points.ply", ws / "scene.json"):  # start clean; stale DBs confuse
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    sparse.mkdir()

    # 1. Features ------------------------------------------------------------
    colmap.run_gpu_aware("feature_extractor", "extract", [
        "feature_extractor", "--database_path", str(db), "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1", "--ImageReader.camera_model", cfg.camera_model,
    ], cfg.cpu_only)
    feats = cm.read_feature_stats(db)
    weak = feats.below(MIN_KEYPOINTS_WARN)
    if feats.median < MIN_KEYPOINTS_MEDIAN_FATAL:
        raise TooFewFeaturesError(
            f"Too few features: median {feats.median} keypoints per image "
            f"(need hundreds to thousands).", hint=TEXTURE_HINT,
            log_path=ws / "logs" / "feature_extractor.log")
    if weak:
        log.warning("%d image(s) have < %d keypoints and will probably fail to register: %s",
                    len(weak), MIN_KEYPOINTS_WARN, ", ".join(weak))
        log.warning(TEXTURE_HINT)

    # 2. Matching ------------------------------------------------------------
    colmap.run_gpu_aware("exhaustive_matcher", "match",
                         ["exhaustive_matcher", "--database_path", str(db)], cfg.cpu_only)
    m = cm.read_match_stats(db)
    if m.largest < 3:
        raise InsufficientOverlapError(
            f"Too few matched features between photos: no image is verified-matched with more "
            f"than one other ({m.pair_count} matched pair(s) across {len(images)} images).",
            hint="Consecutive photos must share most of what they see. Move the camera in smaller "
                 "steps (<= ~15-20 degrees around the object), keep the object at a similar size "
                 "in frame, and avoid blurry frames. If overlap is fine, the object is probably "
                 "too textureless: " + TEXTURE_HINT,
            log_path=ws / "logs" / "exhaustive_matcher.log")
    if len(m.components) > 1:
        log.warning("Images split into %d disconnected groups (sizes %s); only the largest "
                    "can be reconstructed. There is a gap in coverage between groups.",
                    len(m.components), [len(c) for c in m.components])
    if m.isolated:
        log.warning("%d image(s) match nothing else (no overlap): %s",
                    len(m.isolated), ", ".join(m.isolated))

    # 3. Mapper --------------------------------------------------------------
    try:
        out = colmap.run("mapper", ["mapper", "--database_path", str(db),
                                    "--image_path", str(image_dir), "--output_path", str(sparse)])
    except ColmapCommandError as e:
        raise SparseReconstructionError(
            f"COLMAP mapper crashed.\n{e.message}",
            hint="See the log. If it says 'Check failed', the database may be corrupt; rerun.",
            log_path=e.log_path) from e

    best = cm.find_best_model(sparse)
    if best is None:
        why = ("It could not find an initial image pair with enough parallax."
               if "No good initial image pair" in out else "It produced no model.")
        raise SparseReconstructionError(
            f"Sparse reconstruction failed. {why}",
            hint="Usually too little parallax or overlap between views: the camera must actually "
                 "move around the object (not just turn on the spot), in small steps, with the "
                 "object filling a decent part of the frame. " + TEXTURE_HINT,
            log_path=ws / "logs" / "mapper.log")
    n_models = cm.count_models(sparse)
    if n_models > 1:
        log.warning("Mapper produced %d separate models (coverage gaps); using %s (%d images).",
                    n_models, best.path.name, len(best.registered))
    if len(best.registered) < 3:
        raise SparseReconstructionError(
            f"Only {len(best.registered)} image(s) registered.",
            hint="Not enough to triangulate. See overlap/texture advice above.",
            log_path=ws / "logs" / "mapper.log")
    frac = len(best.registered) / len(images)
    if frac < MIN_REGISTERED_FRACTION:
        unreg = sorted({p.name for p in images} - set(best.registered))
        log.warning("Only %d/%d images registered (%.0f%%). Unregistered: %s",
                    len(best.registered), len(images), 100 * frac, ", ".join(unreg))
    else:
        log.info("Registered %d/%d images, %d sparse points.",
                 len(best.registered), len(images), best.num_points)

    # Keep the chosen model at a fixed location for the later stages.
    chosen = sparse / "best"
    shutil.copytree(best.path, chosen)
    result.registered_images, result.sparse_points = len(best.registered), best.num_points

    # 4. Is the camera path good enough for a 3D model? ----------------------
    cov = analyze_camera_coverage(best.registered, best.centers, best.up_dirs, best.points)
    result.coverage_deg = cov.coverage_deg
    (ws / "scene.json").write_text(json.dumps({
        "up": cov.up.tolist(), "center": cov.center.tolist(),
        "coverage_deg": cov.coverage_deg, "largest_gap_deg": cov.largest_gap_deg,
        "linearity": cov.linearity, "elevation_span_deg": cov.elevation_span_deg}, indent=2))
    _check_coverage(cov, cfg.allow_partial, ws)

    try:  # For eyeballing in MeshLab/CloudCompare; not essential.
        colmap.run("model_converter", ["model_converter", "--input_path", str(chosen),
                                       "--output_path", str(ws / "sparse_points.ply"),
                                       "--output_type", "PLY"])
    except ColmapCommandError:
        log.warning("Could not export sparse_points.ply (non-fatal).")


def _check_coverage(cov: CoverageReport, allow_partial: bool, ws: Path) -> None:
    """Degenerate camera geometry produces garbage (or nothing) downstream, so say so now."""
    suffix = " (--allow-partial continues anyway)"
    problem = None
    if cov.linearity < LINEARITY_FATAL:
        problem = DegenerateReconstructionError(
            f"Degenerate camera path: the {cov.n_cameras} solved camera positions lie almost on a "
            f"straight line (linearity {cov.linearity:.3f}). A 3D model needs views from all around "
            f"the object, not from along one side.",
            hint="Move the camera in a circle AROUND the object (full 360 degrees, 20-30 photos), "
                 "keeping it centred in frame, rather than sliding past it." + suffix)
    elif cov.coverage_deg < COVERAGE_FATAL_DEG:
        a, b = cov.gap_between or ("?", "?")
        problem = DegenerateReconstructionError(
            f"Not enough angular coverage: the solved cameras only span {cov.coverage_deg:.0f} "
            f"degrees around the object (largest empty gap {cov.largest_gap_deg:.0f} degrees, "
            f"between {a} and {b}).",
            hint="Go all the way around the object (360 degrees). If you did, the photos in the "
                 "missing stretch failed to register: retake them with more overlap, less blur "
                 "and more texture." + suffix)
    if problem is not None:
        if allow_partial:
            log.warning("%s", problem.message)
        else:
            problem.log_path = ws / "logs" / "mapper.log"
            raise problem
    elif cov.coverage_deg < COVERAGE_WARN_DEG:
        a, b = cov.gap_between or ("?", "?")
        log.warning("Cameras cover %.0f of 360 degrees; the biggest gap (%.0f degrees) is between "
                    "%s and %s. Surfaces facing the gap will be missing or poor.",
                    cov.coverage_deg, cov.largest_gap_deg, a, b)
    if cov.elevation_span_deg < ELEVATION_WARN_DEG:
        log.warning("All shots are at almost the same height (spread %.0f degrees), so the top of "
                    "the object will be missing or poor. Add a few shots from higher up.",
                    cov.elevation_span_deg)


def _dense(cfg, colmap: cm.Colmap, image_dir: Path, ws: Path, result: ReconstructionResult):
    log.info("== Dense reconstruction (backend: %s) ==", cfg.dense_backend)
    model = ws / "sparse" / "best"
    if not (model / "images.bin").exists():
        raise DenseReconstructionError(
            f"No sparse model at {model}.", hint="Run the sparse stage first (--only sparse).")
    if cfg.dense_backend != "auto" and cfg.dense_backend not in BACKENDS:
        raise ReconstructionError(f"Unknown dense backend '{cfg.dense_backend}'.",
                                  hint="Use auto, colmap or openmvs.")

    ctx = DenseContext(colmap, image_dir, model, ws / "dense", ws / "logs", cfg.verbose,
                       cfg.openmvs_bin_dir)
    names = AUTO_ORDER if cfg.dense_backend == "auto" else (cfg.dense_backend,)
    problems: list[tuple[str, ReconstructionError]] = []
    for name in names:
        backend = BACKENDS[name]()
        try:
            backend.check_available(ctx)
            log.info("Using dense backend: %s", name)
            backend.run(ctx)
        except (DenseUnavailableError, OpenMVSNotFoundError) as e:
            if len(names) == 1:
                raise
            log.warning("Dense backend '%s' can't run here: %s", name, e.message)
            problems.append((name, e))
            continue
        result.dense_backend = name
        break
    else:
        raise DenseUnavailableError(
            "No dense backend could run:\n" + "\n".join(f"    - {n}: {e.message}" for n, e in problems),
            hint="Either install OpenMVS for CPU dense reconstruction (README > Installing OpenMVS), "
                 "or use a machine with an NVIDIA GPU and CUDA-enabled COLMAP. The sparse stage "
                 "already succeeded; rerun with --only dense once one is available.")

    n = cm.ply_vertex_count(ctx.fused_ply) if ctx.fused_ply.exists() else 0
    result.dense_points = n
    if n < MIN_DENSE_POINTS:
        raise DenseReconstructionError(
            f"Dense cloud has only {n} points.",
            hint="Depth maps failed to agree across views. Needs more overlap, sharper images "
                 "and texture; see the advice in the sparse stage.",
            log_path=ws / "logs")
    log.info("Dense cloud: %d points.", n)


def _preview(ws: Path) -> Path | None:
    from .preview import render_preview
    mesh = ws / "mesh.ply"
    if not mesh.exists():
        raise ReconstructionError(f"No mesh at {mesh} to preview.", hint="Run the mesh stage first.")
    try:
        out = render_preview(mesh, ws / "preview.png")
        log.info("Preview: %s", out)
        return out
    except Exception as e:  # a failed preview must never cost you the mesh
        log.warning("Could not render preview (%s); the mesh itself is fine.", e)
        return None


def _write_report(ws: Path, result: ReconstructionResult) -> None:
    """Merge into report.json so staged runs (--only ...) accumulate rather than overwrite."""
    path = ws / "report.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report.update({k: v for k, v in asdict(result).items() if v is not None})
    path.write_text(json.dumps(report, indent=2, default=str))
