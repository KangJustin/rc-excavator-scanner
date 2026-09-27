"""Things a real can-on-a-table test run needed: lumpy scans, dark photos, and not typing session names."""
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

o3d = pytest.importorskip("open3d")

from cad.__main__ import main as cad_main
from cad.build import find_freecadcmd
from measure import MeasureError
from measure.scale import scale_from_known
from reconstruct.paths import resolve_latest
from reconstruct.preflight import preflight
from tests.scenes import UNIT_MM, make_scene

needs_freecad = pytest.mark.skipif(find_freecadcmd() is None, reason="FreeCAD not installed")


def lumpy(height_units=1.0, short_sides=(0.5, 0.5)):
    """An `analyze()`-shaped result for a scan whose shape fit is poor (the real Coke can scan)."""
    return {"primitive_fit": {"likely_shape": "neither (irregular or a composite shape)", "fit_quality": "poor"},
            "bounding_box": {"height_above_base_plane_mm": height_units,
                             "oriented_mm": {"length": height_units, "width": short_sides[0], "height": short_sides[1]}}}


# ---- scale from the standing height when the fit is poor ---------------------------------------
def test_poor_fit_scales_from_standing_height_and_checks_the_width():
    scale, info = scale_from_known(lumpy(1.02, (0.5136, 0.5125)), diameter_mm=66, height_mm=122)
    assert scale == pytest.approx(122 / 1.02)
    chk = info["independent_check"]["width"]
    assert chk["measured_mm"] == pytest.approx((0.5136 + 0.5125) / 2 * scale, abs=0.05)
    assert chk["error_percent"] == pytest.approx(100 * (chk["measured_mm"] - 66) / 66, abs=0.05)
    assert "standing height" in info["source"] and "poor" in info["source"]


def test_poor_fit_with_only_a_diameter_is_refused_with_the_way_out():
    with pytest.raises(MeasureError, match="cylinder or sphere") as e:
        scale_from_known(lumpy(), diameter_mm=66)
    assert "--known-height-mm" in str(e.value)


def test_poor_fit_without_a_table_cannot_use_standing_height():
    r = lumpy()
    r["bounding_box"]["height_above_base_plane_mm"] = None
    with pytest.raises(MeasureError):
        scale_from_known(r, height_mm=122)


# ---- cad: known sizes, colour, inferred cylinder --------------------------------------------------
@pytest.fixture(scope="module")
def scenes(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("can_scenes")
    return {k: make_scene(k, tmp)[0] for k in ("can", "box", "tower")}


def run(ws, out, *args):
    rc = cad_main([str(ws), "--out", str(out), *args])
    return rc, json.loads((Path(out) / "cad_report.json").read_text())


@needs_freecad
def test_use_known_sizes_builds_the_ideal_shape_and_still_reports_the_scans_deviation(scenes, tmp_path):
    rc, r = run(scenes["can"], tmp_path / "o", "--scale-mm", str(UNIT_MM), "--known-diameter-mm", "70",
                "--known-height-mm", "130", "--use-known-sizes")
    assert rc == 0 and r["shape"] == "cylinder"
    assert r["params_mm"]["DIAMETER_MM"] == 70 and r["params_mm"]["HEIGHT_MM"] == 130  # NOT the scan's 66 x 122
    assert "you gave" in r["params_source"]
    assert r["build"]["volume_mm3"] == pytest.approx(np.pi * 35 ** 2 * 130, rel=0.005)
    assert r["deviation_from_scan"]["rms_mm"] > 1.0  # 70 x 130 vs a 66 x 122 scan: the deviation is reported, not hidden


@needs_freecad
def test_a_lumpy_scan_with_both_sizes_is_assumed_to_be_a_cylinder_and_says_so(scenes, tmp_path):
    rc, r = run(scenes["tower"], tmp_path / "t", "--scale-mm", str(UNIT_MM), "--known-diameter-mm", "40",
                "--known-height-mm", "70", "--use-known-sizes")
    assert rc == 0 and r["shape"] == "cylinder" and r["detected"] == "neither"
    assert any("cylinder was assumed" in w for w in r["warnings"])


@needs_freecad
def test_use_known_sizes_refuses_boxes(scenes, capsys):
    assert cad_main([str(scenes["box"]), "--scale-mm", str(UNIT_MM), "--known-diameter-mm", "60", "--use-known-sizes"]) == 2
    assert "sphere or cylinder" in capsys.readouterr().err


@needs_freecad
def test_colour_override_wins_and_bad_colours_are_rejected(scenes, tmp_path, capsys):
    rc, r = run(scenes["can"], tmp_path / "c", "--scale-mm", str(UNIT_MM), "--color", "#c8102e")
    assert r["appearance"]["dominant_rgb"] == [200, 16, 46] and r["appearance"]["palette"][0]["label"].startswith("set by you")
    assert cad_main([str(scenes["can"]), "--scale-mm", str(UNIT_MM), "--color", "red"]) == 2
    assert "#c8102e" in capsys.readouterr().err


@needs_freecad
def test_very_dark_scans_get_a_warning_pointing_at_the_photos(scenes, tmp_path):
    ws = scenes["can"]
    dark = ws / "mesh.ply"
    m = o3d.io.read_triangle_mesh(str(dark))
    m.vertex_colors = o3d.utility.Vector3dVector(np.tile([0.22, 0.03, 0.04], (len(m.vertices), 1)))  # a dim maroon
    darkws = tmp_path / "dark"
    darkws.mkdir()
    o3d.io.write_triangle_mesh(str(darkws / "mesh.ply"), m)
    (darkws / "mesh.stl").write_bytes((ws / "mesh.stl").read_bytes())
    (darkws / "report.json").write_text((ws / "report.json").read_text())
    rc, r = run(darkws, tmp_path / "d", "--scale-mm", str(UNIT_MM))
    assert any("very dark" in w and "underexposed" in w and "--color" in w for w in r["warnings"])


# ---- dark photos are flagged before the long reconstruction --------------------------------------
def test_dark_photos_are_flagged_at_preflight(tmp_path, caplog):
    rng = np.random.default_rng(0)
    for i, level in enumerate([50] * 5):
        cv2.imwrite(str(tmp_path / f"{i:04d}.jpg"), np.clip(rng.normal(level, 25, (240, 320, 3)), 0, 255).astype(np.uint8))
    with caplog.at_level(logging.WARNING):
        preflight(tmp_path)
    assert "photos are dark" in caplog.text and "Add light" in caplog.text


def test_well_lit_photos_are_not_flagged(tmp_path, caplog):
    rng = np.random.default_rng(0)
    for i in range(5):
        cv2.imwrite(str(tmp_path / f"{i:04d}.jpg"), np.clip(rng.normal(130, 25, (240, 320, 3)), 0, 255).astype(np.uint8))
    with caplog.at_level(logging.WARNING):
        preflight(tmp_path)
    assert "photos are dark" not in caplog.text


# ---- 'latest' ------------------------------------------------------------------------------------
def test_latest_picks_the_newest_session_that_has_the_right_files(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    for name in ("2026-01-01_1000", "2026-01-02_1000", "empty_one"):
        (tmp_path / "captures" / name).mkdir(parents=True)
    for name in ("2026-01-01_1000", "2026-01-02_1000"):
        (tmp_path / "captures" / name / "0001.jpg").write_bytes(b"x")
        time.sleep(0.02)
    (tmp_path / "captures" / "empty_one").touch()  # newest by mtime, but holds no photos
    assert resolve_latest(Path("latest"), "captures", ("*.jpg",)).name == "2026-01-02_1000"
    assert resolve_latest(Path("captures/x"), "captures", ("*.jpg",)) == Path("captures/x")  # anything else untouched
    with pytest.raises(FileNotFoundError, match="nothing found"):
        resolve_latest(Path("latest"), "reconstructions", ("mesh.stl",))
