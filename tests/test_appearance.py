import struct

import numpy as np
import pytest

from cad.appearance import dominant_colors, vertex_colors_for
from cad.guistate import appearance_blob, gui_document_xml

o3d = pytest.importorskip("open3d")


def shaded(rgb, n, rng, spread=0.25):
    """One colour under varying light: the same hue at different brightness."""
    return np.clip(np.array(rgb) * rng.uniform(1 - spread, 1 + spread, (n, 1)), 0, 1)


def test_one_hue_in_varying_light_is_one_colour_not_three():
    """Regression: clustering raw RGB split an orange ball into 'dark / mid / light orange' and picked the dark one."""
    rng = np.random.default_rng(0)
    pal = dominant_colors(shaded((0.85, 0.42, 0.26), 3000, rng))
    r, g, b = pal[0]["rgb"]
    assert (r, g, b) == pytest.approx((217, 107, 66), abs=14)  # about the true colour, not a shadowed variant
    assert r > g > b


def test_two_real_colours_report_the_bigger_one_as_the_base():
    rng = np.random.default_rng(1)
    red, silver = shaded((0.75, 0.10, 0.10), 2400, rng, 0.15), shaded((0.72, 0.72, 0.74), 600, rng, 0.15)
    pal = dominant_colors(np.vstack([red, silver]))
    assert pal[0]["rgb"][0] > 1.8 * pal[0]["rgb"][2] and pal[0]["share"] >= 0.5  # red body wins (80% of the surface)
    assert any(abs(c["rgb"][0] - c["rgb"][2]) < 25 for c in pal)  # ...and the silver shows up in the palette


def test_highlights_and_shadows_do_not_drag_the_colour():
    rng = np.random.default_rng(2)
    body = shaded((0.30, 0.55, 0.30), 2000, rng, 0.1)
    glare, shadow = np.full((200, 3), 0.99), np.full((200, 3), 0.01)
    r, g, b = dominant_colors(np.vstack([body, glare, shadow]))[0]["rgb"]
    assert g > r and g > b and g == pytest.approx(140, abs=25)  # still green; not washed out or darkened


def test_colours_are_looked_up_from_the_coloured_ply_and_mismatches_are_refused(tmp_path):
    pts = np.random.default_rng(3).uniform(0, 1, (400, 3))
    cols = np.c_[np.linspace(0, 1, 400), np.zeros(400), 1 - np.linspace(0, 1, 400)]
    pc = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(pts), o3d.utility.Vector3iVector([[0, 1, 2]]))
    pc.vertex_colors = o3d.utility.Vector3dVector(cols)
    ply = tmp_path / "mesh.ply"
    o3d.io.write_triangle_mesh(str(ply), pc)
    got = vertex_colors_for(pts[[5, 100, 399]] * 50.0, 50.0, ply)  # the STL side is in mm (scaled by 50)
    assert np.allclose(got, cols[[5, 100, 399]], atol=1e-3)
    assert vertex_colors_for(pts + 10.0, 1.0, ply) is None  # geometry does not line up: don't invent colours
    assert vertex_colors_for(pts, 1.0, tmp_path / "missing.ply") is None


def test_shape_appearance_blob_uses_FreeCADs_blue_first_byte_order():
    """Confirmed in the real GUI: writing R,G,B turned an orange model blue. FreeCAD wants [alpha, B, G, R]."""
    blob = appearance_blob((219, 108, 66))
    assert len(blob) == 40 and struct.unpack("<I", blob[:4])[0] == 1
    assert list(blob[8:12]) == [255, 66, 108, 219]  # diffuse: alpha, B, G, R
    assert struct.unpack("<f", blob[20:24])[0] == pytest.approx(0.9, abs=1e-6)


def test_colour_files_are_referenced_only_for_the_visible_result():
    xml, files = gui_document_xml([("PartDesign::Body", "Body"), ("Sketcher::SketchObject", "S"), ("PartDesign::Pad", "Pad")],
                                  (1, 1, 1), color=(10, 20, 30))
    assert set(files) == {"ShapeAppearance_Body", "ShapeAppearance_Pad"}  # not the hidden sketch
    assert xml.count("<MaterialList ") == 2
    assert gui_document_xml([("PartDesign::Body", "Body")], (1, 1, 1))[1] == {}  # no colour given: nothing added
