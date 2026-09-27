"""Dense point-cloud backends. Both consume the sparse model from the sparse stage and
must leave a point cloud at <workspace>/dense/fused.ply, so the meshing stage doesn't
care which one ran.

    ColmapDense   COLMAP patch_match_stereo + stereo_fusion. Needs an NVIDIA GPU + CUDA build.
    OpenMVSDense  OpenMVS InterfaceCOLMAP + DensifyPointCloud. Runs on CPU (slower).

To add a backend: subclass DenseBackend, implement check_available() and run(), and
register it in BACKENDS.
"""
from __future__ import annotations

import logging
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .colmap import Colmap, run_command
from .errors import (
    ColmapCommandError, DenseReconstructionError, DenseUnavailableError, OpenMVSNotFoundError,
)

log = logging.getLogger(__name__)


@dataclass
class DenseContext:
    colmap: Colmap
    image_dir: Path
    model_dir: Path  # sparse model to densify (…/sparse/best)
    dense_dir: Path  # workspace/dense
    log_dir: Path
    verbose: bool = False
    openmvs_bin_dir: str = ""

    @property
    def fused_ply(self) -> Path:
        return self.dense_dir / "fused.ply"


class DenseBackend(ABC):
    name: str

    @abstractmethod
    def check_available(self, ctx: DenseContext) -> None:
        """Raise DenseUnavailableError / OpenMVSNotFoundError if this backend can't run here."""

    @abstractmethod
    def run(self, ctx: DenseContext) -> Path:
        """Produce ctx.fused_ply and return it."""


def _undistort(ctx: DenseContext) -> None:
    """Both backends start from COLMAP's undistorted images + sparse model."""
    if ctx.dense_dir.exists():
        shutil.rmtree(ctx.dense_dir)
    ctx.colmap.run("image_undistorter", [
        "image_undistorter", "--image_path", str(ctx.image_dir), "--input_path", str(ctx.model_dir),
        "--output_path", str(ctx.dense_dir), "--output_type", "COLMAP"])


class ColmapDense(DenseBackend):
    name = "colmap"

    def check_available(self, ctx: DenseContext) -> None:
        _, cuda = ctx.colmap.version_and_cuda()
        if cuda is False:
            raise DenseUnavailableError(
                "This COLMAP build has no CUDA, and COLMAP's dense stereo (patch_match_stereo) "
                "needs a GPU backend. macOS builds, including every Apple Silicon Mac, have none.",
                hint="Use OpenMVS instead (--dense-backend openmvs; CPU, slower), or run the dense "
                     "stage on a machine with an NVIDIA GPU and a CUDA-enabled COLMAP "
                     "(README > Splitting stages across machines).")

    def run(self, ctx: DenseContext) -> Path:
        _undistort(ctx)
        try:
            ctx.colmap.run("patch_match_stereo", [
                "patch_match_stereo", "--workspace_path", str(ctx.dense_dir),
                "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "true"])
        except ColmapCommandError as e:
            low = e.output.lower()
            if "cuda" in low:
                raise DenseUnavailableError(
                    "COLMAP's patch_match_stereo needs CUDA and an NVIDIA GPU.",
                    hint="Use --dense-backend openmvs (CPU), or run this stage on a CUDA machine.",
                    log_path=e.log_path) from e
            if "out of memory" in low:
                raise DenseReconstructionError(
                    "GPU ran out of memory in patch_match_stereo.",
                    hint="Capture at a lower resolution, or use --dense-backend openmvs.",
                    log_path=e.log_path) from e
            raise DenseReconstructionError(e.message, log_path=e.log_path) from e
        ctx.colmap.run("stereo_fusion", [
            "stereo_fusion", "--workspace_path", str(ctx.dense_dir), "--workspace_format", "COLMAP",
            "--input_type", "geometric", "--output_path", str(ctx.fused_ply)])
        return ctx.fused_ply


OPENMVS_TOOLS = ("InterfaceCOLMAP", "DensifyPointCloud")


def find_openmvs_tool(name: str, bin_dir: str = "") -> str | None:
    """Look in the configured folder first, then on PATH."""
    if bin_dir:
        p = Path(bin_dir).expanduser() / name
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return shutil.which(name)


class OpenMVSDense(DenseBackend):
    name = "openmvs"

    def check_available(self, ctx: DenseContext) -> None:
        missing = [t for t in OPENMVS_TOOLS if not find_openmvs_tool(t, ctx.openmvs_bin_dir)]
        if missing:
            where = f"'{ctx.openmvs_bin_dir}' and PATH" if ctx.openmvs_bin_dir else "PATH"
            raise OpenMVSNotFoundError(
                f"OpenMVS tool(s) not found on {where}: {', '.join(missing)}.",
                hint="OpenMVS has no Homebrew formula: build it from source (README > Installing "
                     "OpenMVS), then point at its bin folder with --openmvs-bin-dir, the "
                     "OPENMVS_BIN_DIR environment variable, or OPENMVS_BIN_DIR in reconstruct/config.py.")

    def run(self, ctx: DenseContext) -> Path:
        interface = find_openmvs_tool("InterfaceCOLMAP", ctx.openmvs_bin_dir)
        densify = find_openmvs_tool("DensifyPointCloud", ctx.openmvs_bin_dir)
        _undistort(ctx)  # -> dense/{images,sparse}: the layout InterfaceCOLMAP expects
        d = ctx.dense_dir.resolve()
        scene, dense_scene = d / "scene.mvs", d / "scene_dense.mvs"
        try:
            run_command([interface, "-i", str(d), "-o", str(scene), "--image-folder", str(d / "images"),
                         "-w", str(d)], "openmvs_interface", ctx.log_dir, ctx.verbose, "OpenMVS", cwd=d)
            # Writes scene_dense.mvs and scene_dense.ply next to it.
            run_command([densify, "-i", str(scene), "-o", str(dense_scene), "-w", str(d)],
                        "openmvs_densify", ctx.log_dir, ctx.verbose, "OpenMVS", cwd=d)
        except ColmapCommandError as e:
            if "out of memory" in e.output.lower() or "bad_alloc" in e.output.lower():
                raise DenseReconstructionError(
                    "OpenMVS ran out of memory.",
                    hint="Capture at a lower resolution or close other apps.", log_path=e.log_path) from e
            raise DenseReconstructionError(e.message, log_path=e.log_path) from e

        ply = d / "scene_dense.ply"
        if not ply.exists():
            raise DenseReconstructionError(
                f"OpenMVS finished but wrote no {ply.name}.",
                hint=f"Look in {ctx.log_dir}/openmvs_densify.log; low overlap/texture can yield an empty cloud.")
        shutil.copyfile(ply, ctx.fused_ply)
        return ctx.fused_ply


BACKENDS: dict[str, type[DenseBackend]] = {"colmap": ColmapDense, "openmvs": OpenMVSDense}
AUTO_ORDER = ("colmap", "openmvs")
