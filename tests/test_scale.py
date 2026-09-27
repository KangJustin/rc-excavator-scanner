import json

import pytest

o3d = pytest.importorskip("open3d")

from cad.__main__ import main as cad_main
from measure import MeasureError, analyze
from measure.__main__ import main as measure_main
from measure.scale import scale_from_known
from tests.scenes import UNIT_MM, make_scene


@pytest.fixture(scope="module")
def scenes(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("scale_scenes")
    return {k: make_scene(k, tmp)[0] for k in ("can", "ball", "box")}


def unscaled(ws):
    return analyze(ws / "mesh.stl", None, z_up=True)


def test_diameter_sets_the_scale_and_height_is_an_independent_check(scenes):
    """The synthetic can is 66 mm x 122 mm with 1 model unit = 50 mm. Give only the diameter: the height must come out right."""
    scale, info = scale_from_known(unscaled(scenes["can"]), diameter_mm=66.0, height_mm=122.0)
    assert scale == pytest.approx(UNIT_MM, rel=0.02)
    chk = info["independent_check"]["height"]
    assert abs(chk["error_percent"]) < 2.0 and chk["expected_mm"] == 122.0
    assert "diameter" not in info.get("independent_check", {})  # the value used for scale is not "checked" against itself


def test_height_alone_can_set_the_scale(scenes):
    scale, info = scale_from_known(unscaled(scenes["can"]), height_mm=122.0)
    assert scale == pytest.approx(UNIT_MM, rel=0.02) and info["source"].startswith("known height")


def test_a_wrong_known_size_is_exposed_by_the_check(scenes):
    """If you type the wrong diameter the height check disagrees loudly. That is the point of giving two sizes."""
    _, info = scale_from_known(unscaled(scenes["can"]), diameter_mm=80.0, height_mm=122.0)
    assert info["independent_check"]["height"]["error_percent"] > 15


def test_sphere_scales_from_its_diameter(scenes):
    scale, info = scale_from_known(unscaled(scenes["ball"]), diameter_mm=80.0)
    assert scale == pytest.approx(UNIT_MM, rel=0.03) and info["shape"] == "sphere"
    with pytest.raises(MeasureError, match="no separate height"):
        scale_from_known(unscaled(scenes["ball"]), height_mm=80.0)


def test_a_box_is_refused_with_a_useful_hint(scenes):
    with pytest.raises(MeasureError, match="cylinder or sphere") as e:
        scale_from_known(unscaled(scenes["box"]), diameter_mm=60.0)
    assert "--scale-mm" in str(e.value)
    with pytest.raises(MeasureError, match="Give"):
        scale_from_known(unscaled(scenes["can"]))


def test_measure_cli_records_how_the_scale_was_derived(scenes, capsys):
    ws = scenes["can"]
    assert measure_main([str(ws), "--known-diameter-mm", "66", "--known-height-mm", "122"]) == 0
    out = capsys.readouterr().out
    assert "Scale from known diameter = 66 mm" in out and "independent check, height" in out
    data = json.loads((ws / "measurements.json").read_text())
    assert data["scale"]["provided"] is True and data["scale"]["source"] == "known diameter = 66 mm"
    assert abs(data["scale"]["independent_check"]["height"]["error_percent"]) < 2
    assert data["primitive_fit"]["cylinder"]["diameter_mm"] == pytest.approx(66, rel=0.001)  # by construction


def test_scale_options_are_mutually_exclusive(scenes, capsys):
    assert measure_main([str(scenes["can"]), "--scale-mm", "50", "--known-diameter-mm", "66"]) == 2
    assert "not both" in capsys.readouterr().err


def test_cad_cli_accepts_the_known_size_too(scenes, capsys):
    from cad.build import find_freecadcmd
    if find_freecadcmd() is None:
        pytest.skip("FreeCAD not installed")
    ws = scenes["can"]
    assert cad_main([str(ws), "--known-diameter-mm", "66", "--known-height-mm", "122", "--out", str(ws / "c2")]) == 0
    out = capsys.readouterr().out
    assert "independent check, height" in out and "cylinder, parametric" in out
    r = json.loads((ws / "c2" / "cad_report.json").read_text())
    assert r["params_mm"]["DIAMETER_MM"] == pytest.approx(66, rel=0.005) and r["params_mm"]["HEIGHT_MM"] == pytest.approx(122, rel=0.03)
