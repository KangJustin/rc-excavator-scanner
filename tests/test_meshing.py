import numpy as np
import pytest

o3d = pytest.importorskip("open3d")

from reconstruct.meshing import MeshOptions, build_mesh


def sphere_cloud(n=20000, r=1.0, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(v * r)
    pcd.normals = o3d.utility.Vector3dVector(v)
    return pcd


def test_sphere_to_ply_and_stl(tmp_path):
    fused = tmp_path / "dense" / "fused.ply"
    fused.parent.mkdir()
    o3d.io.write_point_cloud(str(fused), sphere_cloud())
    res = build_mesh(fused, tmp_path, MeshOptions(poisson_depth=7, density_trim_quantile=0.0))
    assert res.ply.stat().st_size > 0 and res.stl.stat().st_size > 0
    assert all(e == pytest.approx(2.0, rel=0.1) for e in res.extents)
    assert res.watertight
    assert len(o3d.io.read_triangle_mesh(str(res.stl)).triangles) == res.triangles


def test_remove_plane_and_scale(tmp_path):
    rng = np.random.default_rng(1)
    sph = np.asarray(sphere_cloud(15000, 1.0).points) + [0, 0, 1.0]  # sits on z=0 table
    table = np.c_[rng.uniform(-4, 4, (20000, 2)), np.zeros(20000)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack([sph, table]))
    fused = tmp_path / "dense" / "fused.ply"
    fused.parent.mkdir()
    o3d.io.write_point_cloud(str(fused), pcd)
    res = build_mesh(fused, tmp_path, MeshOptions(poisson_depth=7, remove_plane=True, scale=10))
    assert max(res.extents) == pytest.approx(20, rel=0.2)  # just the ball, x10, not the 8 m table


def test_too_few_points_is_a_clear_error(tmp_path):
    from reconstruct.errors import MeshingError
    fused = tmp_path / "f.ply"
    o3d.io.write_point_cloud(str(fused), sphere_cloud(200))
    with pytest.raises(MeshingError, match="only 200 points"):
        build_mesh(fused, tmp_path)
