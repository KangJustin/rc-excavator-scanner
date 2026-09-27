"""A real failure: a shiny bowl did not reconstruct, leaving only the floor, and the tool built a 66 x 122 mm 'can' from it."""
import pytest

o3d = pytest.importorskip("open3d")

from cad.__main__ import main as cad_main
from measure import MeasureError, analyze
from measure.__main__ import main as measure_main
from measure.scale import scale_from_known
from tests.scenes import UNIT_MM, make_scene


@pytest.fixture(scope="module")
def scenes(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("noobj")
    return {k: make_scene(k, tmp)[0] for k in ("empty", "can")}


def test_a_bare_floor_is_reported_as_a_flat_surface_not_a_cylinder(scenes):
    """Regression: a very wide, very short cylinder fits a flat sheet perfectly ('cylinder, fit good')."""
    r = analyze(scenes["empty"] / "mesh.stl", UNIT_MM, z_up=True)
    pf = r["primitive_fit"]
    assert pf["likely_shape"].startswith("flat surface") and pf["fit_quality"] == "poor"
    assert any("MISSING" in w and "flat sheet" in w for w in r["warnings"])
    assert "flat surface" in r["summary_text"]


def test_scaling_from_a_flat_floor_is_refused_with_advice(scenes):
    unscaled = analyze(scenes["empty"] / "mesh.stl", None, z_up=True)
    with pytest.raises(MeasureError, match="No object found") as e:
        scale_from_known(unscaled, diameter_mm=66, height_mm=122)
    assert "matte or textured" in str(e.value)


def test_cad_refuses_to_build_from_a_bare_floor_even_with_known_sizes(scenes, capsys):
    assert cad_main([str(scenes["empty"]), "--known-diameter-mm", "66", "--known-height-mm", "122", "--use-known-sizes"]) == 2
    assert "No object found" in capsys.readouterr().err
    assert cad_main([str(scenes["empty"]), "--scale-mm", str(UNIT_MM)]) == 2  # plain auto mode too
    assert "No object found" in capsys.readouterr().err


def test_sizes_that_disagree_with_the_scan_are_a_hard_stop_unless_overridden(scenes):
    """The synthetic can is 66 x 122. Claim 100 mm across: the implied height is ~185 mm, far from the 122 you also said."""
    unscaled = analyze(scenes["can"] / "mesh.stl", None, z_up=True)
    with pytest.raises(MeasureError, match="disagree") as e:
        scale_from_known(unscaled, diameter_mm=100, height_mm=122)
    assert "preview.png" in str(e.value) and "--allow-mismatch" in str(e.value)
    scale, info = scale_from_known(unscaled, diameter_mm=100, height_mm=122, allow_mismatch=True)  # explicit override
    assert abs(info["independent_check"]["height"]["error_percent"]) > 25


def test_small_disagreements_still_pass(scenes):
    unscaled = analyze(scenes["can"] / "mesh.stl", None, z_up=True)
    scale, info = scale_from_known(unscaled, diameter_mm=66, height_mm=126)  # 3% off
    assert abs(info["independent_check"]["height"]["error_percent"]) < 10


def test_cli_stops_on_mismatch_and_allow_flag_continues(scenes, capsys):
    ws = str(scenes["can"])
    assert measure_main([ws, "--known-diameter-mm", "100", "--known-height-mm", "122"]) == 2
    assert "disagree" in capsys.readouterr().err
    assert measure_main([ws, "--known-diameter-mm", "100", "--known-height-mm", "122", "--allow-mismatch"]) == 0
