import json
import logging
import stat
import sys
from pathlib import Path

import cv2
import pytest

from capture import CaptureSession, FakeCamera
from reconstruct import ReconstructionConfig, run_reconstruction
from reconstruct.errors import (
    DegenerateReconstructionError, DenseReconstructionError, DenseUnavailableError,
    InsufficientOverlapError, OpenMVSNotFoundError, PreflightError, ReconstructionError,
    SparseReconstructionError, TooFewFeaturesError,
)
from reconstruct.meshing import MeshOptions

HERE = Path(__file__).parent


def make_wrapper(path: Path, script: str, *lead):
    path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{HERE / script}" {" ".join(lead)} "$@"\n')
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # default output goes to ./reconstructions/<name>


@pytest.fixture
def images(tmp_path):
    s = CaptureSession(tmp_path / "captures", "2026-09-25_2130")
    cam = FakeCamera()
    for _ in range(8):
        s.save_frame(cam.read())
    return s.dir


@pytest.fixture
def colmap_bin(tmp_path):
    return make_wrapper(tmp_path / "colmap", "fake_colmap.py")


@pytest.fixture
def openmvs_dir(tmp_path):
    d = tmp_path / "openmvs_bin"
    d.mkdir()
    make_wrapper(d / "InterfaceCOLMAP", "fake_openmvs.py", "InterfaceCOLMAP")
    make_wrapper(d / "DensifyPointCloud", "fake_openmvs.py", "DensifyPointCloud")
    return str(d)


def cfg(images, colmap_bin, **kw):
    kw.setdefault("openmvs_bin_dir", "")
    return ReconstructionConfig(images, colmap_bin=colmap_bin, mesh=MeshOptions(poisson_depth=7), **kw)


# --- happy paths -------------------------------------------------------------
def test_full_pipeline_with_colmap_backend(images, colmap_bin, tmp_path):
    r = run_reconstruction(cfg(images, colmap_bin))
    ws = r.workspace
    assert ws == tmp_path / "reconstructions" / "2026-09-25_2130"  # outside the image folder
    assert r.registered_images == 8 and r.dense_points == 20000 and r.dense_backend == "colmap"
    assert r.mesh.stl.stat().st_size > 0 and r.mesh.ply.stat().st_size > 0
    assert r.mesh.aligned and r.mesh.extents[2] == pytest.approx(1.0, rel=0.15)  # r=0.5 ball, Z-up
    assert cv2.imread(str(r.preview)) is not None
    scene = json.loads((ws / "scene.json").read_text())
    assert scene["coverage_deg"] > 300 and scene["up"][2] > 0.9
    assert (ws / "logs" / "mapper.log").exists() and (ws / "report.json").exists()


def test_stages_run_separately_and_report_accumulates(images, colmap_bin):
    a = run_reconstruction(cfg(images, colmap_bin, only="sparse"))
    assert a.mesh is None
    run_reconstruction(cfg(images, colmap_bin, only="dense"))
    run_reconstruction(cfg(images, colmap_bin, only="mesh", preview=False))
    assert not (a.workspace / "preview.png").exists()
    run_reconstruction(cfg(images, colmap_bin, only="preview"))
    assert (a.workspace / "preview.png").exists()
    rep = json.loads((a.workspace / "report.json").read_text())
    assert rep["registered_images"] == 8 and rep["dense_points"] == 20000 and "mesh" in rep


def test_openmvs_backend_end_to_end(images, colmap_bin, openmvs_dir):
    r = run_reconstruction(cfg(images, colmap_bin, dense_backend="openmvs", openmvs_bin_dir=openmvs_dir))
    assert r.dense_backend == "openmvs" and r.dense_points == 20000
    assert (r.workspace / "dense" / "fused.ply").exists()  # same hand-off file as the COLMAP backend
    assert (r.workspace / "logs" / "openmvs_densify.log").exists()
    assert r.mesh.stl.stat().st_size > 0


# --- backend selection / CPU fallback ---------------------------------------
def test_auto_falls_back_to_openmvs_when_colmap_has_no_cuda(images, colmap_bin, openmvs_dir, monkeypatch, caplog):
    monkeypatch.setenv("FAKE_CUDA", "0")
    with caplog.at_level(logging.WARNING):
        r = run_reconstruction(cfg(images, colmap_bin, openmvs_bin_dir=openmvs_dir))
    assert r.dense_backend == "openmvs" and "colmap" in caplog.text


def test_auto_falls_back_even_when_cuda_only_fails_at_runtime(images, colmap_bin, openmvs_dir, monkeypatch):
    monkeypatch.setenv("FAKE_CUDA", "unknown")  # header doesn't say; patch_match_stereo then fails
    r = run_reconstruction(cfg(images, colmap_bin, openmvs_bin_dir=openmvs_dir))
    assert r.dense_backend == "openmvs"


def test_auto_with_neither_backend_explains_both_and_keeps_sparse(images, colmap_bin, monkeypatch):
    monkeypatch.setenv("FAKE_CUDA", "0")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")  # no OpenMVS anywhere
    with pytest.raises(DenseUnavailableError) as e:
        run_reconstruction(cfg(images, colmap_bin))
    msg = str(e.value)
    assert "colmap:" in msg and "openmvs:" in msg and "Installing OpenMVS" in msg
    assert (Path("reconstructions") / "2026-09-25_2130" / "sparse_points.ply").exists()


def test_explicit_colmap_backend_without_cuda_points_at_openmvs(images, colmap_bin, monkeypatch):
    monkeypatch.setenv("FAKE_CUDA", "0")
    with pytest.raises(DenseUnavailableError) as e:
        run_reconstruction(cfg(images, colmap_bin, dense_backend="colmap"))
    assert "Apple Silicon" in str(e.value) and "--dense-backend openmvs" in str(e.value)


def test_explicit_openmvs_backend_missing_tools(images, colmap_bin, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    with pytest.raises(OpenMVSNotFoundError) as e:
        run_reconstruction(cfg(images, colmap_bin, dense_backend="openmvs", openmvs_bin_dir="/nope"))
    assert "InterfaceCOLMAP" in str(e.value) and "OPENMVS_BIN_DIR" in str(e.value) and "'/nope' and PATH" in str(e.value)


def test_openmvs_crash_is_reported_not_swallowed(images, colmap_bin, openmvs_dir, monkeypatch):
    monkeypatch.setenv("FAKE_OPENMVS", "fail")
    with pytest.raises(DenseReconstructionError, match="memory"):
        run_reconstruction(cfg(images, colmap_bin, dense_backend="openmvs", openmvs_bin_dir=openmvs_dir))


def test_unknown_backend_name(images, colmap_bin):
    with pytest.raises(ReconstructionError, match="Unknown dense backend"):
        run_reconstruction(cfg(images, colmap_bin, dense_backend="magic"))


# --- failure modes -----------------------------------------------------------
def test_too_few_features(images, colmap_bin, monkeypatch):
    monkeypatch.setenv("FAKE_KP", "40")
    with pytest.raises(TooFewFeaturesError) as e:
        run_reconstruction(cfg(images, colmap_bin))
    assert "median 40" in str(e.value) and "matte" in str(e.value)


def test_insufficient_overlap_means_too_few_matches(images, colmap_bin, monkeypatch):
    monkeypatch.setenv("FAKE_PAIRS", "none")
    with pytest.raises(InsufficientOverlapError) as e:
        run_reconstruction(cfg(images, colmap_bin))
    assert "Too few matched features" in str(e.value) and "smaller steps" in str(e.value)


def test_mapper_finds_no_model(images, colmap_bin, monkeypatch):
    monkeypatch.setenv("FAKE_MAPPER", "fail")
    with pytest.raises(SparseReconstructionError, match="initial image pair"):
        run_reconstruction(cfg(images, colmap_bin))


def test_straight_line_camera_path_is_degenerate(images, colmap_bin, monkeypatch):
    monkeypatch.setenv("FAKE_POSES", "line")
    with pytest.raises(DegenerateReconstructionError) as e:
        run_reconstruction(cfg(images, colmap_bin))
    assert "straight line" in str(e.value) and "AROUND the object" in str(e.value)
    assert "--allow-partial" in str(e.value)


def test_short_arc_is_insufficient_angular_coverage(images, colmap_bin, monkeypatch):
    monkeypatch.setenv("FAKE_POSES", "arc")
    with pytest.raises(DegenerateReconstructionError) as e:
        run_reconstruction(cfg(images, colmap_bin))
    assert "angular coverage" in str(e.value) and "degrees" in str(e.value)


def test_allow_partial_downgrades_to_warning(images, colmap_bin, monkeypatch, caplog):
    monkeypatch.setenv("FAKE_POSES", "arc")
    with caplog.at_level(logging.WARNING):
        r = run_reconstruction(cfg(images, colmap_bin, allow_partial=True))
    assert r.mesh is not None and "angular coverage" in caplog.text


def test_flat_orbit_warns_about_missing_top(images, colmap_bin, monkeypatch, caplog):
    monkeypatch.setenv("FAKE_POSES", "flat")
    with caplog.at_level(logging.WARNING):
        r = run_reconstruction(cfg(images, colmap_bin, only="sparse"))
    assert r.registered_images == 8 and "same height" in caplog.text


def test_too_few_images(tmp_path, colmap_bin):
    s = CaptureSession(tmp_path / "c", "s")
    s.save_frame(FakeCamera().read())
    with pytest.raises(PreflightError, match="at least 3"):
        run_reconstruction(cfg(s.dir, colmap_bin))


def test_missing_image_dir_and_missing_colmap_leave_nothing_behind(tmp_path, images):
    with pytest.raises(PreflightError, match="not found"):
        run_reconstruction(cfg(tmp_path / "nope", "colmap"))
    with pytest.raises(ReconstructionError, match="not found on PATH"):
        run_reconstruction(cfg(images, "definitely-not-colmap"))
    assert not (tmp_path / "reconstructions").exists()


def test_output_inside_image_folder_is_refused(images, colmap_bin):
    c = cfg(images, colmap_bin)
    c.out_dir = images / "reconstruction"
    with pytest.raises(PreflightError, match="inside the image folder"):
        run_reconstruction(c)
