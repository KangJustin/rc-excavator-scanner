import re
import zipfile

import pytest

from cad.guistate import _objects, add_gui_state, gui_document_xml

OBJS = [("Spreadsheet::Sheet", "Params"), ("PartDesign::Body", "Body"), ("App::Origin", "Origin"),
        ("App::Plane", "XY_Plane"), ("Sketcher::SketchObject", "ProfileSketch"), ("PartDesign::Pad", "Pad")]


def visible(xml):
    out = {}
    for n in re.findall(r'<ViewProvider name="(\w+)"', xml):
        block = re.search(r'<ViewProvider name="%s".*?</ViewProvider>' % n, xml, re.S).group(0)
        out[n] = re.search(r'Bool value="(\w+)"', block).group(1) == "true"
    return out


def test_body_and_its_tip_feature_are_shown_and_everything_else_hidden():
    """A PartDesign Body draws its TIP feature: if that is hidden the 3D view is empty (the bug this fixes)."""
    v = visible(gui_document_xml(OBJS, (10, 10, 20))[0])
    assert v == {"Params": False, "Body": True, "Origin": False, "XY_Plane": False, "ProfileSketch": False, "Pad": True}


def test_solid_fallback_object_is_shown():
    v = visible(gui_document_xml([("Part::Feature", "ScannedSolid")], (1, 1, 1))[0])
    assert v == {"ScannedSolid": True}


def test_camera_is_isometric_and_scales_with_the_model():
    small = gui_document_xml(OBJS, (10, 10, 10))[0]
    big = gui_document_xml(OBJS, (100, 100, 100))[0]
    h = lambda x: float(re.search(r"height ([\d.]+)", x).group(1))
    assert h(big) == pytest.approx(10 * h(small)) and "orientation 0.74290609" in small


def test_object_list_parses_document_xml_order():
    xml = '<Objects Count="2"><Object type="A::B" name="One" id="1" /><Object type="C::D" name="Two" id="2" /></Objects><ObjectData><Object name="Zed"/>'
    assert _objects(xml) == [("A::B", "One"), ("C::D", "Two")]


def test_add_gui_state_rewrites_the_zip_keeps_everything_and_is_idempotent(tmp_path):
    f = tmp_path / "m.FCStd"
    doc = ('<Document><Objects Count="2"><Object type="PartDesign::Body" name="Body" id="1" />'
           '<Object type="PartDesign::Pad" name="Pad" id="2" /></Objects><ObjectData/></Document>')
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("Document.xml", doc)
        z.writestr("Pad.Shape.brp", b"\x00\x01binary")
    assert add_gui_state(f, (5, 5, 5)) is True
    with zipfile.ZipFile(f) as z:
        assert z.testzip() is None and z.read("Pad.Shape.brp") == b"\x00\x01binary" and z.read("Document.xml").decode() == doc
        assert visible(z.read("GuiDocument.xml").decode()) == {"Body": True, "Pad": True}
    assert add_gui_state(f, (5, 5, 5)) is False  # never overwrites a GUI-saved file's real state
