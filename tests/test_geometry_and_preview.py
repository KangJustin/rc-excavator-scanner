import cv2
import numpy as np
import pytest

from reconstruct.geometry import analyze_camera_coverage
from tests.fake_colmap import camera_centres, look_at_pose

o3d = pytest.importorskip("open3d")


def solve(mode, n=24):
    C = camera_centres(n, mode)
    ups = np.array([-look_at_pose(c)[0][1] for c in C])
    pts = np.random.default_rng(0).normal(size=(300, 3)) * 0.3
    return analyze_camera_coverage([f"{i}.jpg" for i in range(n)], np.array(C), ups, pts)


def test_full_orbit_is_fine():
    r = solve("circle")
    assert r.coverage_deg > 330 and r.linearity > 0.5 and r.elevation_span_deg > 8
    assert abs(r.up[2]) > 0.9 and r.up[2] > 0  # up is +Z, signed by the cameras' own up


def test_arc_has_limited_coverage_and_names_the_gap():
    r = solve("arc")
    assert 70 < r.coverage_deg < 90 and r.largest_gap_deg > 270
    assert r.gap_between[0] != r.gap_between[1]


def test_straight_line_is_flagged_as_linear():
    assert solve("line").linearity < 0.06


def test_flat_orbit_has_no_elevation_spread():
    assert solve("flat").elevation_span_deg < 8


def sphere_mesh(tmp_path):
    from reconstruct.meshing import MeshOptions, build_mesh
    rng = np.random.default_rng(0)
    v = rng.normal(size=(20000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(v)
    pcd.normals = o3d.utility.Vector3dVector(v)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(0.5 + 0.5 * v, 0, 1))
    (tmp_path / "dense").mkdir()
    o3d.io.write_point_cloud(str(tmp_path / "dense" / "fused.ply"), pcd)
    return build_mesh(tmp_path / "dense" / "fused.ply", tmp_path, MeshOptions(poisson_depth=7))


def test_preview_sheet_renders_six_views_of_the_object(tmp_path):
    from reconstruct.preview import render_preview
    res = sphere_mesh(tmp_path)
    png = render_preview(res.ply, tmp_path / "preview.png", size=200)
    img = cv2.imread(str(png))
    assert img.shape == (400, 600, 3)  # 3 x 2 grid
    tile = img[:200, :200]
    background = np.array([245, 245, 245])
    covered = np.mean(np.any(np.abs(tile.astype(int) - background) > 12, axis=2))
    assert 0.25 < covered < 0.75  # a sphere fills ~ pi*0.45^2 of its tile, not blank and not everything


def test_alignment_stands_the_object_up_along_the_orbit_normal(tmp_path):
    from reconstruct.meshing import MeshOptions, build_mesh
    rng = np.random.default_rng(0)
    v = rng.normal(size=(30000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v = v * [0.3, 1.0, 0.3]  # tall along Y
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(v)
    n = v / np.array([0.3, 1.0, 0.3]) ** 2
    pcd.normals = o3d.utility.Vector3dVector(n / np.linalg.norm(n, axis=1, keepdims=True))
    (tmp_path / "dense").mkdir()
    o3d.io.write_point_cloud(str(tmp_path / "dense" / "fused.ply"), pcd)
    res = build_mesh(tmp_path / "dense" / "fused.ply", tmp_path, MeshOptions(poisson_depth=7),
                     scene={"up": [0, 1, 0]})
    w, d, h = res.extents
    assert res.aligned and h == pytest.approx(2.0, rel=0.1) and w == pytest.approx(0.6, rel=0.15)
    m = o3d.io.read_triangle_mesh(str(res.ply))
    assert np.asarray(m.vertices)[:, 2].min() == pytest.approx(0.0, abs=1e-6)
    res2 = build_mesh(tmp_path / "dense" / "fused.ply", tmp_path,
                      MeshOptions(poisson_depth=7, align_up=False), scene={"up": [0, 1, 0]})
    assert not res2.aligned
