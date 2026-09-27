import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")

from cad.__main__ import main
from cad.build import _consistently_oriented, close_object, find_freecadcmd
from cad.plan import make_plan, rotation_between
from cad.verify import deviation, to_cad_frame
from tests.scenes import UNIT_MM, make_scene

needs_freecad = pytest.mark.skipif(find_freecadcmd() is None, reason="FreeCAD (freecadcmd) not installed")


# ---- pure geometry (no FreeCAD) ----------------------------------------------------------------
def test_rotation_between_maps_vectors_including_opposites():
    for a, b in [((0, 0, 1), (1, 0, 0)), ((0, 0, 1), (0.3, -0.5, 0.8)), ((0, 0, 1), (0, 0, -1))]:
        R = rotation_between(a, b)
        b = np.array(b, float) / np.linalg.norm(b)
        assert np.allclose(R @ np.array(a, float), b) and np.allclose(R.T @ R, np.eye(3)) and np.linalg.det(R) > 0


def open_box(w=10.0, d=6.0, h=4.0):
    """A box with its BOTTOM missing, like an object cut off a table. Z from `lift` up, so it also floats a bit."""
    V = np.array([[x, y, z] for z in (h * 0.2, h) for y in (0, d) for x in (0, w)], float)
    idx = lambda x, y, z: x + 2 * y + 4 * z
    quads = [((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),  # top (wound outward, like the sides)
             ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)),
             ((1, 1, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)), ((0, 1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1))]
    F = []
    for q in quads:
        a, b, c, e = (idx(*p) for p in q)
        F += [[a, b, c], [a, c, e]]
    return V, np.array(F)


def test_orientation_checker_catches_a_flipped_patch():
    """Regression: a mesh with a few reversed triangles looks closed (and Open3D's is_orientable said so)
    but gives origin-dependent volumes. Our own check must catch it."""
    V, F = open_box()
    Vc, Fc, ok = close_object(V, F, None)
    assert ok and _consistently_oriented(Fc)
    bad = Fc.copy()
    bad[0] = bad[0][::-1]
    assert not _consistently_oriented(bad)


def test_closing_gives_an_outward_solid_with_the_exact_volume_and_reaches_the_plane():
    V, F = open_box(10, 6, 4)  # bottom at z=0.8, top at z=4; the "table" plane is z=0
    plane = {"removed": True, "plane_normal": [0, 0, 1], "plane_offset": 0.0, "tolerance": 0.5}
    Vc, Fc, ok = close_object(V, F, plane)
    assert ok
    t = Vc[Fc] - Vc.mean(axis=0)
    vol = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6
    assert vol == pytest.approx(10 * 6 * 4, rel=1e-6)  # positive (outward) and it includes the 0.8 mm gap to the plane
    assert Vc[:, 2].min() == pytest.approx(0.0, abs=1e-9)
    for origin in (np.zeros(3), np.array([100.0, -50.0, 20.0])):  # volume must not depend on the reference point
        tt = Vc[Fc] - origin
        assert np.einsum("ij,ij->i", tt[:, 0], np.cross(tt[:, 1], tt[:, 2])).sum() / 6 == pytest.approx(vol, rel=1e-6)


def test_deviation_is_zero_for_the_same_surface_and_scales_with_offset(tmp_path):
    V, F = open_box()
    Vc, Fc, _ = close_object(V, F, None)
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Vc), o3d.utility.Vector3iVector(Fc))
    o3d.io.write_triangle_mesh(str(tmp_path / "a.stl"), m)  # STL needs normals
    m.compute_triangle_normals()
    o3d.io.write_triangle_mesh(str(tmp_path / "a.stl"), m)
    d0 = deviation(Vc, Fc, np.eye(4), tmp_path / "a.stl")
    assert d0["rms_mm"] < 0.01 and d0["grade"] == "excellent"
    shift = np.eye(4); shift[:3, 3] = [0.7, 0, 0]  # CAD frame shifted 0.7 mm from the scan
    assert 0.2 < deviation(Vc, Fc, shift, tmp_path / "a.stl")["rms_mm"] < 0.7  # only faces facing X are off by the full 0.7
    assert np.allclose(to_cad_frame(np.array([[1.0, 2, 3]]), shift), [[0.3, 2, 3]])


# ---- whole tool, in real headless FreeCAD ----------------------------------------------------
@pytest.fixture(scope="module")
def scenes(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cad_scenes")
    return {k: make_scene(k, tmp) for k in ("can", "ball", "box", "tower")}


def run(ws, *extra):
    rc = main([str(ws), *extra])
    rep = json.loads((Path(ws) / "cad" / "cad_report.json").read_text())
    return rc, rep


@needs_freecad
def test_can_becomes_a_parametric_cylinder_that_matches_the_scan(scenes):
    ws, truth = scenes["can"]
    rc, r = run(ws, "--scale-mm", str(UNIT_MM))
    assert rc == 0 and r["shape"] == "cylinder" and r["parametric"] and r["build"]["valid"]
    assert r["params_mm"]["DIAMETER_MM"] == pytest.approx(66, rel=0.03) and r["params_mm"]["HEIGHT_MM"] == pytest.approx(122, rel=0.03)
    assert r["build"]["volume_mm3"] == pytest.approx(truth["volume_mm3"], rel=0.03)
    assert r["deviation_from_scan"]["grade"] == "excellent" and r["deviation_from_scan"]["rms_mm"] < 0.5
    feats = r["build"]["features"]
    assert "PartDesign::Pad" in feats and "Sketcher::SketchObject" in feats and "Spreadsheet::Sheet" in feats
    for k in ("fcstd", "step", "stl", "macro", "preview"):
        assert Path(r["files"][k]).stat().st_size > 100, k
    import zipfile  # opens showing the model in the GUI, not an empty view
    assert "GuiDocument.xml" in zipfile.ZipFile(r["files"]["fcstd"]).namelist()
    # the synthetic scenes carry vertex colours in mesh.ply, so the model is coloured and an appearance picture exists
    assert "appearance" in r and len(r["appearance"]["dominant_rgb"]) == 3
    assert any(n.startswith("ShapeAppearance_") for n in zipfile.ZipFile(r["files"]["fcstd"]).namelist())
    assert Path(r["files"]["appearance"]).stat().st_size > 1000 and Path(r["files"]["scan_colored_ply"]).stat().st_size > 1000


@needs_freecad
def test_box_becomes_a_parametric_box(scenes):
    ws, truth = scenes["box"]
    rc, r = run(ws, "--scale-mm", str(UNIT_MM))
    p = r["params_mm"]
    assert rc == 0 and r["shape"] == "box" and r["parametric"]
    assert sorted([p["LENGTH_MM"], p["WIDTH_MM"], p["HEIGHT_MM"]]) == pytest.approx([60, 60, 80], rel=0.03)
    assert p["HEIGHT_MM"] == pytest.approx(80, rel=0.03)  # the vertical side is used as the height
    assert r["deviation_from_scan"]["grade"] == "excellent"


@needs_freecad
def test_ball_becomes_a_sphere(scenes):
    ws, truth = scenes["ball"]
    rc, r = run(ws, "--scale-mm", str(UNIT_MM))
    assert r["shape"] == "sphere" and r["params_mm"]["DIAMETER_MM"] == pytest.approx(80, rel=0.03)
    assert r["build"]["volume_mm3"] == pytest.approx(truth["volume_mm3"], rel=0.05)


@needs_freecad
def test_irregular_object_falls_back_to_an_exact_nonparametric_solid(scenes):
    """A block with a tower matches no primitive. It must become a valid solid of the right volume that
    stands on the plane, and must say plainly that it is NOT parametric."""
    ws, truth = scenes["tower"]
    rc, r = run(ws, "--scale-mm", str(UNIT_MM))
    assert rc == 0 and r["shape"] == "solid" and r["parametric"] is False and r["detected"] == "neither"
    assert r["build"]["valid"] and r["build"]["volume_mm3"] == pytest.approx(truth["volume_mm3"], rel=0.03)
    assert r["build"]["bbox_mm"][2] == pytest.approx(truth["height_mm"], rel=0.03)  # reaches the plane: full height
    assert r["deviation_from_scan"]["rms_mm"] < 0.5
    assert not any("PartDesign::Pad" in f for f in r["build"]["features"])
    assert "NOT parametric" in Path(r["files"]["macro"]).read_text()
    assert Path(r["files"]["step"]).read_text(errors="ignore").startswith("ISO-10303")


@needs_freecad
def test_forcing_a_shape_and_the_saved_scale(scenes, tmp_path):
    ws, _ = scenes["can"]
    # measure saves its scale; cad must pick it up without --scale-mm
    from measure.__main__ import main as measure_main
    assert measure_main([str(ws), "--scale-mm", str(UNIT_MM)]) == 0
    rc = main([str(ws), "--shape", "solid", "--out", str(tmp_path / "out")])
    r = json.loads((tmp_path / "out" / "cad_report.json").read_text())  # --out sends everything to that folder
    assert rc == 0 and r["shape"] == "solid" and r["detected"] == "cylinder"
    assert r["build"]["valid"] and r["parametric"] is False


def test_refuses_to_guess_a_scale(scenes, capsys):
    ws, _ = scenes["ball"]
    (ws / "measurements.json").unlink(missing_ok=True)
    assert main([str(ws)]) == 2
    assert "No scale" in capsys.readouterr().err


@needs_freecad
def test_generated_macro_is_standalone_and_editable(scenes, tmp_path):
    """The macro written for the user must run on its own in FreeCAD and stay parametric."""
    ws, _ = scenes["can"]
    run(ws, "--scale-mm", str(UNIT_MM))
    macro = (ws / "cad" / "can_table.FCMacro").read_text()
    assert "DIAMETER_MM =" in macro and "HEIGHT_MM =" in macro and "Params" in macro
    script = tmp_path / "s.py"
    script.write_text(macro + """
doc = App.ActiveDocument
b, sh = doc.Body, doc.Params
v0 = b.Shape.Volume
sh.set("B2", "=100 mm"); doc.recompute()   # change Height
print("RESULT", v0, b.Shape.Volume, b.Shape.BoundBox.ZLength)
""")
    out = subprocess.run([find_freecadcmd(), str(script)], capture_output=True, text=True, timeout=120)
    line = [l for l in (out.stdout + out.stderr).splitlines() if "RESULT" in l][-1].split()
    v0, v1, z = float(line[1]), float(line[2]), float(line[3])
    assert z == pytest.approx(100) and v1 == pytest.approx(v0 * 100 / 122.6, rel=0.02)


def test_ear_clipping_triangulates_a_concave_polygon_exactly():
    from cad.build import _ear_clip
    L = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 3), (0, 3)]  # an L: area 4*1 + 1*2 = 6
    tris = _ear_clip(np.array(L, float), list(range(6)))
    area = sum(abs((L[b][0] - L[a][0]) * (L[c][1] - L[a][1]) - (L[c][0] - L[a][0]) * (L[b][1] - L[a][1])) / 2 for a, b, c in tris)
    assert len(tris) == 4 and area == pytest.approx(6.0)
    assert all(((L[b][0] - L[a][0]) * (L[c][1] - L[a][1]) - (L[c][0] - L[a][0]) * (L[b][1] - L[a][1])) > 0 for a, b, c in tris)  # same winding as the polygon
