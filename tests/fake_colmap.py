"""A stand-in for the COLMAP CLI that writes the same files/tables real COLMAP would,
so the pipeline's orchestration and error handling can be tested without COLMAP.

Behaviour switches (env vars):
  FAKE_CUDA=0/1        header says with/without CUDA; without => patch_match_stereo fails
  FAKE_CUDA=unknown    header says nothing about CUDA, but patch_match_stereo still fails
  FAKE_KP=<n>          keypoints per image
  FAKE_PAIRS=none      no verified matches
  FAKE_MAPPER=fail     mapper finds no model
  FAKE_POSES=circle|arc|line|flat   camera path in the solved model (default circle, 2 heights)
"""
import os
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

MOD = 2147483647


def opt(args, name):
    return args[args.index(name) + 1]


def rot_to_quat(R):
    """Rotation matrix -> (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        return np.array([s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1) * 2
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = s / 4
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def camera_centres(n, mode):
    if mode == "line":
        return [np.array([-3 + 6 * i / max(n - 1, 1), -4.0, 1.0]) for i in range(n)]
    span = {"arc": np.radians(80), "circle": 2 * np.pi, "flat": 2 * np.pi}[mode]
    step = span / n if mode != "arc" else span / max(n - 1, 1)
    out = []
    for i in range(n):
        h = 1.0 + (1.5 if (mode == "circle" and i % 4 == 0) else 0.0)  # a few higher shots
        out.append(np.array([4 * np.cos(i * step), 4 * np.sin(i * step), h]))
    return out


def look_at_pose(c, target=np.zeros(3)):
    f = target - c
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    R = np.stack([right, down, f])  # world -> camera (COLMAP: x right, y down, z forward)
    return R, -R @ c


def write_model(d, names, mode):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(names)))
        for i, (n, c) in enumerate(zip(names, camera_centres(len(names), mode))):
            R, t = look_at_pose(c)
            q = rot_to_quat(R)
            f.write(struct.pack("<i4d3di", i + 1, *q, *t, 1) + n.encode() + b"\0" + struct.pack("<Q", 0))
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(500, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * 0.5
    with open(d / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for i, p in enumerate(pts):
            f.write(struct.pack("<Q3d3BdQ", i + 1, *p, 128, 128, 128, 0.5, 0))
    (d / "cameras.bin").write_bytes(b"\0" * 8)


def write_sphere_ply(path, n=20000):
    rng = np.random.default_rng(0)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v = v * 0.5
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
                "property float nx\nproperty float ny\nproperty float nz\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p in v:
            f.write(f"{p[0]} {p[1]} {p[2]} {p[0]} {p[1]} {p[2]} {int(128 + 100 * p[0])} 150 {int(128 + 100 * p[2])}\n")


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "help":
        cuda = os.environ.get("FAKE_CUDA", "1")
        tail = {"1": " with CUDA", "0": " without CUDA"}.get(cuda, "")
        print(f"COLMAP 3.9.1 -- Structure-from-Motion and Multi-View Stereo\n(Commit fake on 2024-01-01{tail})")
    elif cmd == "feature_extractor":
        db, imgs = Path(opt(args, "--database_path")), sorted(Path(opt(args, "--image_path")).glob("*.jpg"))
        assert opt(args, "--ImageReader.single_camera") == "1"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT)")
        con.execute("CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)")
        con.execute("CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER)")
        for i, p in enumerate(imgs, 1):
            con.execute("INSERT INTO images VALUES (?,?)", (i, p.name))
            con.execute("INSERT INTO keypoints VALUES (?,?,?,NULL)", (i, int(os.environ.get("FAKE_KP", 2000)), 128))
        con.commit(); con.close()
    elif cmd == "exhaustive_matcher":
        con = sqlite3.connect(opt(args, "--database_path"))
        n = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        if os.environ.get("FAKE_PAIRS") != "none":
            for i in range(1, n):
                con.execute("INSERT INTO two_view_geometries VALUES (?,?)", (i * MOD + i + 1, 100))
        con.commit(); con.close()
    elif cmd == "mapper":
        if os.environ.get("FAKE_MAPPER") == "fail":
            print("E0101 mapper.cc: No good initial image pair found."); return
        con = sqlite3.connect(opt(args, "--database_path"))
        names = [r[0] for r in con.execute("SELECT name FROM images ORDER BY image_id")]
        write_model(Path(opt(args, "--output_path")) / "0", names, os.environ.get("FAKE_POSES", "circle"))
    elif cmd == "model_converter":
        Path(opt(args, "--output_path")).write_text("ply\n")
    elif cmd == "image_undistorter":
        src, out = Path(opt(args, "--input_path")), Path(opt(args, "--output_path"))
        assert (src / "images.bin").exists()
        (out / "images").mkdir(parents=True)
        (out / "sparse").mkdir()
        for n in ("cameras.bin", "images.bin", "points3D.bin"):
            (out / "sparse" / n).write_bytes((src / n).read_bytes())
    elif cmd == "patch_match_stereo":
        assert Path(opt(args, "--workspace_path"), "images").is_dir()
        if os.environ.get("FAKE_CUDA", "1") in ("0", "unknown"):
            print("Dense stereo reconstruction requires CUDA, which is not available."); sys.exit(1)
    elif cmd == "stereo_fusion":
        write_sphere_ply(Path(opt(args, "--output_path")))
    else:
        print(f"fake colmap: unsupported command {cmd}"); sys.exit(2)


if __name__ == "__main__":
    main()
