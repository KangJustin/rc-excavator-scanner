"""COLMAP itself isn't needed: we build the same SQLite tables / binary files it writes."""
import sqlite3

import numpy as np

from reconstruct import colmap as cm

MOD = 2147483647


def make_db(path, names, kp, pairs):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT)")
    con.execute("CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)")
    con.execute("CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER)")
    for i, n in enumerate(names, 1):
        con.execute("INSERT INTO images VALUES (?,?)", (i, n))
        con.execute("INSERT INTO keypoints VALUES (?,?,?,NULL)", (i, kp[i - 1], 128))
    for a, b, n in pairs:
        con.execute("INSERT INTO two_view_geometries VALUES (?,?)", (a * MOD + b, n))
    con.commit(); con.close()


def test_feature_stats(tmp_path):
    db = tmp_path / "d.db"
    make_db(db, ["a.jpg", "b.jpg", "c.jpg"], [2000, 50, 900], [])
    f = cm.read_feature_stats(db)
    assert f.median == 900 and f.below(300) == ["b.jpg"]


def test_match_graph_finds_gaps_and_isolated(tmp_path):
    db = tmp_path / "d.db"
    names = [f"{i}.jpg" for i in range(6)]
    # 1-2-3 connected, 4-5 connected (gap between groups), 6 isolated, one weak pair ignored
    make_db(db, names, [500] * 6, [(1, 2, 80), (2, 3, 60), (4, 5, 40), (3, 4, 5)])
    m = cm.read_match_stats(db)
    assert [len(c) for c in m.components] == [3, 2]
    assert m.isolated == ["5.jpg"]
    assert m.largest == 3


from tests.fake_colmap import write_model  # noqa: E402


def test_sparse_model_parsing_and_best_pick(tmp_path):
    write_model(tmp_path / "0", ["a.jpg", "b.jpg", "c.jpg"], "circle")
    write_model(tmp_path / "1", ["d.jpg", "e.jpg"], "circle")
    best = cm.find_best_model(tmp_path)
    assert best.path.name == "0" and best.registered == ["a.jpg", "b.jpg", "c.jpg"]
    assert best.num_points == 500 and best.points.shape == (500, 3)
    assert best.centers.shape == (3, 3) and abs(np.linalg.norm(best.centers[0][:2]) - 4) < 1e-6  # 4 m circle
    assert cm.count_models(tmp_path) == 2


def test_no_model_returns_none(tmp_path):
    (tmp_path / "0").mkdir()
    assert cm.find_best_model(tmp_path) is None


def test_ply_vertex_count(tmp_path):
    p = tmp_path / "x.ply"
    p.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 4242\nend_header\n")
    assert cm.ply_vertex_count(p) == 4242
