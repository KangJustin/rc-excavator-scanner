"""Dense point cloud -> cleaned cloud -> Poisson mesh -> .ply + .stl (Open3D)."""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import MeshingError

log = logging.getLogger(__name__)

MIN_POINTS = 1000
KEEP_PIECE_FRACTION = 0.05  # mesh pieces smaller than this share of the largest piece are discarded


@dataclass
class MeshOptions:
    poisson_depth: int = 9  # 8 = smooth/fast, 10+ = detailed but noisy
    # Open3D 0.20's multithreaded Poisson intermittently aborts the whole process on macOS
    # arm64 (~1 run in 5 in testing, no Python exception to catch). Single-threaded was stable;
    # raise this (-1 = all cores) on a machine where you've verified it's safe.
    poisson_threads: int = 1
    density_trim_quantile: float = 0.02  # drop the least-supported 2% of mesh vertices (0 = keep all)
    outlier_neighbors: int = 30
    outlier_std_ratio: float = 2.0
    voxel_size: float | None = None  # downsample (in model units) before meshing
    crop: tuple[float, float, float, float, float, float] | None = None  # xmin ymin zmin xmax ymax zmax
    remove_plane: bool = False  # RANSAC-remove the dominant plane (the table)
    plane_distance_frac: float = 0.01  # plane thickness as a fraction of the cloud's bbox diagonal
    largest_cluster: bool = False  # keep only the biggest connected blob of points
    scale: float = 1.0  # multiply the mesh (SfM scale is arbitrary; see README)
    align_up: bool = True  # rotate to Z-up using the camera-orbit plane, sit on z=0, centre on x/y


@dataclass
class MeshResult:
    ply: Path
    stl: Path
    points_in: int
    points_used: int
    vertices: int
    triangles: int
    watertight: bool
    extents: tuple[float, float, float]  # bounding-box size in output units
    aligned: bool = False  # True if rotated to Z-up (extents are then width x depth x height)


def rotation_to_z(up: np.ndarray) -> np.ndarray:
    """Rotation matrix taking the direction `up` to +Z."""
    up = np.asarray(up, float)
    up = up / np.linalg.norm(up)
    v = np.cross(up, [0.0, 0.0, 1.0])
    c = float(up[2])
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def _o3d():
    try:
        import open3d as o3d
    except ImportError as e:
        raise MeshingError("Open3D is not installed.", hint="pip install open3d") from e
    return o3d


def clean_point_cloud(pcd, opts: MeshOptions):
    o3d = _o3d()
    if opts.crop:
        lo, hi = np.array(opts.crop[:3]), np.array(opts.crop[3:])
        pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(lo, hi))
        log.info("crop -> %d points", len(pcd.points))

    if opts.voxel_size:
        pcd = pcd.voxel_down_sample(opts.voxel_size)
        log.info("voxel downsample -> %d points", len(pcd.points))

    pcd, _ = pcd.remove_statistical_outlier(opts.outlier_neighbors, opts.outlier_std_ratio)
    log.info("outlier removal -> %d points", len(pcd.points))

    if opts.remove_plane and len(pcd.points) > 100:
        diag = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())
        _, inliers = pcd.segment_plane(opts.plane_distance_frac * diag, 3, 1000)
        pcd = pcd.select_by_index(inliers, invert=True)
        log.info("plane removal -> %d points", len(pcd.points))

    if opts.largest_cluster and len(pcd.points) > 100:
        eps = 5 * float(np.mean(pcd.compute_nearest_neighbor_distance()))
        labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=10))
        valid = labels[labels >= 0]
        if len(valid):
            keep = Counter(valid.tolist()).most_common(1)[0][0]
            pcd = pcd.select_by_index(np.where(labels == keep)[0])
        log.info("largest cluster -> %d points", len(pcd.points))
    return pcd


def poisson_mesh(pcd, opts: MeshOptions):
    o3d = _o3d()
    if not pcd.has_normals():
        log.info("No normals in cloud; estimating.")
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
        try:
            pcd.orient_normals_consistent_tangent_plane(30)
        except RuntimeError:  # Open3D throws an opaque error on some degenerate clouds
            log.warning("Normal orientation failed; orienting outward from the centroid "
                        "(fine for compact convex-ish objects, wrong for concave ones).")
            pts, nrm = np.asarray(pcd.points), np.asarray(pcd.normals)
            flip = np.einsum("ij,ij->i", nrm, pts - pts.mean(axis=0)) < 0
            nrm[flip] *= -1
            pcd.normals = o3d.utility.Vector3dVector(nrm)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=opts.poisson_depth, n_threads=opts.poisson_threads)
    if len(mesh.triangles) == 0:
        raise MeshingError(
            "Poisson reconstruction produced an empty mesh.",
            hint="The cloud is probably too sparse or has inconsistent normals. Check "
                 "dense/fused.ply in MeshLab/CloudCompare; re-shoot with more overlap.",
        )
    # Poisson invents surface where it has no data; the low-density vertices are those.
    d = np.asarray(densities)
    mesh.remove_vertices_by_mask(d < np.quantile(d, opts.density_trim_quantile))

    # Discard floating fragments: keep the largest connected component only.
    tri_cluster, n_tri, _ = mesh.cluster_connected_triangles()
    tri_cluster, n_tri = np.asarray(tri_cluster), np.asarray(n_tri)
    if len(n_tri) > 1:
        # Keep every piece that is a meaningful fraction of the largest, not only the largest:
        # an object with a tiny footprint on the table (a ball) is its own piece, and the
        # table would otherwise win and the object vanish. Only small floaters are dropped.
        keep = n_tri >= KEEP_PIECE_FRACTION * n_tri.max()
        mesh.remove_triangles_by_mask(~keep[tri_cluster])
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()  # Poisson leaves a few; slicers choke on them
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()  # required for STL
    return mesh


def build_mesh(fused_ply: Path, out_dir: Path, opts: MeshOptions | None = None,
               scene: dict | None = None) -> MeshResult:
    """`scene` (from the sparse stage's scene.json) supplies the up-vector for alignment."""
    opts = opts or MeshOptions()
    o3d = _o3d()
    if not fused_ply.exists():
        raise MeshingError(f"Point cloud not found: {fused_ply}",
                           hint="Run the dense stage first (--only dense).")
    pcd = o3d.io.read_point_cloud(str(fused_ply))
    n_in = len(pcd.points)
    if n_in < MIN_POINTS:
        raise MeshingError(f"Point cloud has only {n_in} points.",
                           hint="Too little geometry for a surface. See the dense-stage advice in the README.")

    pcd = clean_point_cloud(pcd, opts)
    if len(pcd.points) < MIN_POINTS:
        raise MeshingError(
            f"Only {len(pcd.points)} points left after cleaning.",
            hint="Cropping/plane/cluster filters removed too much. Loosen --crop, or drop "
                 "--remove-plane / --largest-cluster.",
        )
    o3d.io.write_point_cloud(str(out_dir / "cleaned_points.ply"), pcd)

    mesh = poisson_mesh(pcd, opts)
    if opts.scale != 1.0:
        mesh.scale(opts.scale, center=(0, 0, 0))
        mesh.compute_triangle_normals()
    aligned = False
    if opts.align_up and scene and scene.get("up") is not None:
        mesh.rotate(rotation_to_z(np.array(scene["up"])), center=(0, 0, 0))
        bb = mesh.get_axis_aligned_bounding_box()
        ctr, lo = bb.get_center(), bb.get_min_bound()
        mesh.translate((-ctr[0], -ctr[1], -lo[2]))
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        aligned = True
    extents = tuple(float(x) for x in mesh.get_axis_aligned_bounding_box().get_extent())
    # "Closed and manifold" is what a slicer needs. (Not mesh.is_watertight(): that also
    # tests self-intersection, which is very slow on scan-sized meshes.)
    watertight = bool(mesh.is_edge_manifold(False) and mesh.is_vertex_manifold())
    if not watertight:
        log.warning("Mesh is not watertight; most slicers can repair this, but check before printing.")

    ply, stl = out_dir / "mesh.ply", out_dir / "mesh.stl"
    ok_ply = o3d.io.write_triangle_mesh(str(ply), mesh)
    ok_stl = o3d.io.write_triangle_mesh(str(stl), mesh)
    if not (ok_ply and ok_stl and ply.stat().st_size and stl.stat().st_size):
        raise MeshingError("Failed to write mesh files.", hint=f"Check permissions on {out_dir}")

    return MeshResult(ply, stl, n_in, len(pcd.points), len(mesh.vertices),
                      len(mesh.triangles), watertight, extents, aligned)
