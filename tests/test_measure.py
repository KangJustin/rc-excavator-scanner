import json

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")

from measure import MeasureError, analyze
from measure.__main__ import main
from measure.fit import fit_cylinder, fit_sphere, sample_surface
from tests.scenes import UNIT_MM, make_scene


# --- primitive fitting on analytic shapes (fast, exact ground truth) -------------------------
def cyl_mesh(r=33.0, h=120.0, n=64, axis=(0, 0, 1)):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[r * np.cos(th), r * np.sin(th)]
    V = np.vstack([np.c_[ring, np.zeros(n)], np.c_[ring, np.full(n, h)], [[0, 0, 0]], [[0, 0, h]]])
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
    return V, np.array(F)


def test_sample_surface_is_area_uniform():
    V, F = cyl_mesh()
    P, N = sample_surface(V, F, 20000, np.random.default_rng(0))
    assert np.allclose(np.linalg.norm(N, axis=1), 1)
    side = np.mean(np.abs(N[:, 2]) < 0.3)
    area_side = 2 * np.pi * 33 * 120
    area_caps = 2 * np.pi * 33 ** 2
    assert side == pytest.approx(area_side / (area_side + area_caps), abs=0.02)


def test_sphere_fit_recovers_center_radius_and_flags_noise():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(5000, 3))
    P = v / np.linalg.norm(v, axis=1, keepdims=True) * 40 + [10, -5, 20]
    f = fit_sphere(P + rng.normal(scale=0.2, size=P.shape))
    assert f.radius == pytest.approx(40, abs=0.1) and np.allclose(f.center, [10, -5, 20], atol=0.1)
    assert f.rel_rms < 0.01


@pytest.mark.parametrize("axis", [(0, 0, 1), (1, 0, 0), (0.3, -0.5, 0.8)])
def test_cylinder_fit_recovers_any_orientation(axis):
    V, F = cyl_mesh()
    a = np.array(axis, float); a /= np.linalg.norm(a)
    _, _, Vt = np.linalg.svd(np.random.default_rng(0).normal(size=(3, 3)))
    # rotate the z-axis onto `a`
    z = np.array([0, 0, 1.0]); v = np.cross(z, a); c = z @ a
    R = np.eye(3) if np.linalg.norm(v) < 1e-9 else (
        np.eye(3) + np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        + np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) @ np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / (1 + c))
    P, N = sample_surface(V @ R.T, F, 20000, np.random.default_rng(0))
    f = fit_cylinder(P, N)
    assert f.radius == pytest.approx(33, rel=0.02) and f.height == pytest.approx(120, rel=0.02)
    assert abs(f.axis @ a) > 0.999 and f.rel_rms < 0.02


def test_sphere_is_not_mistaken_for_a_good_cylinder():
    rng = np.random.default_rng(2)
    v = rng.normal(size=(8000, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    cyl = fit_cylinder(v * 30, v)
    assert cyl is None or cyl.rel_rms > 0.1


# --- whole pipeline on realistic scenes (Poisson mesh of an object on a noisy table) -----------
@pytest.fixture(scope="module")
def scenes(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("scenes")
    return {(k, t): make_scene(k, tmp, with_table=t) for k in ("can", "ball", "box") for t in (True, False)}


def test_can_on_table_is_measured_to_within_a_couple_percent(scenes):
    ws, truth = scenes[("can", True)]
    r = analyze(ws / "mesh.stl", UNIT_MM, z_up=True)
    assert r["plane_removal"]["removed"] and r["plane_removal"]["plane_tilt_deg"] < 2
    bb = r["bounding_box"]
    assert bb["height_above_base_plane_mm"] == pytest.approx(122, rel=0.02)
    assert bb["axis_aligned_mm"]["z"] == pytest.approx(122, rel=0.02)
    assert sorted(bb["oriented_mm"][k] for k in ("length", "width", "height")) == pytest.approx([66, 66, 122], rel=0.03)
    assert r["volume"]["mm3"] == pytest.approx(truth["volume_mm3"], rel=0.03)
    pf = r["primitive_fit"]
    assert pf["likely_shape"] == "cylinder" and pf["fit_quality"] == "good"
    assert pf["cylinder"]["diameter_mm"] == pytest.approx(66, rel=0.03) and pf["cylinder"]["height_mm"] == pytest.approx(122, rel=0.03)
    assert pf["cylinder"]["axis_tilt_from_vertical_deg"] < 3
    assert r["obb_fill_ratio"] == pytest.approx(0.785, abs=0.04)  # a cylinder fills pi/4 of its box
    assert not r["warnings"]


def test_ball_with_tiny_footprint_is_not_lost(scenes):
    """Regression: the meshing step used to keep only the largest piece, dropping a ball resting on a table."""
    ws, truth = scenes[("ball", True)]
    r = analyze(ws / "mesh.stl", UNIT_MM, z_up=True)
    assert r["plane_removal"]["removed"]
    assert r["primitive_fit"]["likely_shape"] == "sphere" and r["primitive_fit"]["fit_quality"] == "good"
    assert r["primitive_fit"]["sphere"]["diameter_mm"] == pytest.approx(80, rel=0.03)
    # the unseen underside is on the plane, so the box height is the full diameter, not just the visible part
    assert r["bounding_box"]["axis_aligned_mm"]["z"] == pytest.approx(80, rel=0.03)
    assert r["volume"]["mm3"] == pytest.approx(truth["volume_mm3"], rel=0.05)
    assert r["obb_fill_ratio"] == pytest.approx(0.524, abs=0.05)


def test_box_is_recognised_as_a_box_with_its_dimensions(scenes):
    ws, truth = scenes[("box", True)]
    r = analyze(ws / "mesh.stl", UNIT_MM, z_up=True)
    assert r["plane_removal"]["removed"]
    pf = r["primitive_fit"]
    assert pf["likely_shape"] == "box" and pf["fit_quality"] == "good"
    assert pf["sphere"]["rms_relative_to_radius"] > 0.1 and pf["cylinder"]["rms_relative_to_radius"] > 0.1
    assert sorted(pf["box"]["dimensions_mm"]) == pytest.approx([60, 60, 80], rel=0.03)
    assert pf["box"]["volume_mm3"] == pytest.approx(truth["volume_mm3"], rel=0.04)
    assert r["volume"]["mm3"] == pytest.approx(truth["volume_mm3"], rel=0.03)
    assert r["obb_fill_ratio"] > 0.9  # box fills its box
    assert r["bounding_box"]["height_above_base_plane_mm"] == pytest.approx(80, rel=0.02)


@pytest.mark.parametrize("kind", ["can", "ball", "box"])
def test_no_table_skips_plane_removal_gracefully_and_still_measures(scenes, kind):
    ws, truth = scenes[(kind, False)]
    r = analyze(ws / "mesh.stl", UNIT_MM, z_up=True)
    assert r["plane_removal"]["removed"] is False and r["plane_removal"]["reason"]
    assert any("Plane removal skipped" in w for w in r["warnings"])
    assert r["bounding_box"]["oriented_mm"]["length"] > 60  # still measured the whole mesh
    # a box's own top/side face must NOT be mistaken for the table and cut away
    assert r["bounding_box"]["axis_aligned_mm"]["z"] > 0.75 * truth["height_mm"] or kind == "ball"
    if r["volume"]["mm3"] is None:  # open mesh + no plane: refuses to invent a volume, offers the hull bound
        assert r["volume"]["convex_hull_mm3"] > 0 and any("unreliable" in w for w in r["warnings"])


def test_works_on_a_tumbled_mesh_that_is_not_known_to_be_z_up(scenes, tmp_path):
    ws, truth = scenes[("can", True)]
    m = o3d.io.read_triangle_mesh(str(ws / "mesh.stl"))
    m.rotate(m.get_rotation_matrix_from_xyz((0.7, 0.4, 1.1)), center=(0, 0, 0))
    m.compute_triangle_normals()
    o3d.io.write_triangle_mesh(str(tmp_path / "t.stl"), m)
    r = analyze(tmp_path / "t.stl", UNIT_MM, z_up=False)
    assert r["plane_removal"]["removed"]
    assert r["volume"]["mm3"] == pytest.approx(truth["volume_mm3"], rel=0.03)
    assert r["primitive_fit"]["cylinder"]["height_mm"] == pytest.approx(122, rel=0.03)
    assert "not known to be Z-up" in r["bounding_box"]["axis_aligned_mm"]["note"]


def test_no_plane_removal_flag_measures_everything(scenes):
    ws, _ = scenes[("can", True)]
    r = analyze(ws / "mesh.stl", UNIT_MM, plane_removal=False, z_up=True)
    assert r["bounding_box"]["axis_aligned_mm"]["x"] > 250  # the whole 300 mm table is in there


def test_unscaled_is_loudly_flagged_and_scale_is_linear(scenes):
    ws, _ = scenes[("can", True)]
    a = analyze(ws / "mesh.stl", None, z_up=True)
    b = analyze(ws / "mesh.stl", 10.0, z_up=True)
    assert a["scale"]["provided"] is False and "NOT SCALED" in a["summary_text"]
    assert any("--scale-mm" in w for w in a["warnings"]) and b["scale"]["provided"] is True
    ratio = b["bounding_box"]["axis_aligned_mm"]["z"] / a["bounding_box"]["axis_aligned_mm"]["z"]
    assert ratio == pytest.approx(10, rel=0.01)
    assert b["volume"]["mm3"] / a["volume"]["mm3"] == pytest.approx(1000, rel=0.01)  # volume scales cubed


def test_stl_vertex_welding(scenes):
    """STL has no shared vertices; loading must weld them or connectivity-based steps break."""
    from measure.analyze import load_mesh
    ws, _ = scenes[("can", True)]
    raw = o3d.io.read_triangle_mesh(str(ws / "mesh.stl"))
    V, F = load_mesh(ws / "mesh.stl")
    assert len(V) < 0.6 * len(raw.vertices) and len(F) > 1000


# --- CLI ---------------------------------------------------------------------------------------
def test_cli_writes_measurements_json_and_prints_block(scenes, tmp_path, capsys):
    ws, _ = scenes[("can", True)]
    assert main([str(ws), "--scale-mm", str(UNIT_MM)]) == 0
    out = capsys.readouterr().out
    assert "=== MEASUREMENTS:" in out and "likely shape: cylinder" in out and "Saved" in out
    data = json.loads((ws / "measurements.json").read_text())
    assert data["units"] == "mm" and data["primitive_fit"]["likely_shape"] == "cylinder"
    assert data["object"]["z_is_up"] is True  # read from reconstruct's report.json
    assert "MEASUREMENTS" in data["summary_text"]


def test_cli_errors_are_actionable(tmp_path, capsys):
    (tmp_path / "photos").mkdir()
    (tmp_path / "photos" / "0001.jpg").write_bytes(b"x")
    assert main([str(tmp_path / "photos")]) == 2
    assert "folder of photos" in capsys.readouterr().err
    (tmp_path / "empty").mkdir()
    assert main([str(tmp_path / "empty")]) == 2
    assert "python -m reconstruct" in capsys.readouterr().err
    assert main([str(tmp_path / "nope.stl")]) == 2
    with pytest.raises(MeasureError, match="positive"):
        analyze(tmp_path / "x.stl", -1)
