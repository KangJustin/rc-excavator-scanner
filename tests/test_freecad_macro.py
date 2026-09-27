"""Runs freecad/scanned_sphere.FCMacro inside a real (headless) FreeCAD. Skipped if FreeCAD isn't installed."""
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest

MACRO = Path(__file__).parent.parent / "freecad" / "scanned_sphere.FCMacro"
CANDIDATES = ["/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd", "freecadcmd"]
FREECADCMD = next((p for p in (shutil.which(c) or (c if Path(c).exists() else None) for c in CANDIDATES) if p), None)
pytestmark = pytest.mark.skipif(FREECADCMD is None, reason="FreeCAD (freecadcmd) not installed")


def set_var(src, name, value):
    """Override a top-level macro variable whatever its current default is."""
    out, n = re.subn(rf"^{name} = .*$", f"{name} = {value!r}", src, flags=re.M)
    assert n == 1, name
    return out


def run_macro(tmp_path, diameter, resize_to=None, export=None):
    src = set_var(set_var(MACRO.read_text(), "DIAMETER_MM", float(diameter)), "SCALE_CONFIRMED", True)
    if export:
        src = set_var(src, "EXPORT_STL_PATH", str(export))
    script = tmp_path / "run.py"
    script.write_text(src + f'''
import json
doc = App.ActiveDocument
sk, body, sheet = doc.ProfileSketch, doc.Body, doc.Params
out = dict(volume=body.Shape.Volume, valid=body.Shape.isValid(), zlen=body.Shape.BoundBox.ZLength,
           zmin=body.Shape.BoundBox.ZMin, solve=sk.solve(), fully=sk.FullyConstrained,
           expr=[list(e) for e in sk.ExpressionEngine], label=doc.Label,
           tree=[o.TypeId for o in doc.Objects if o.TypeId.startswith(("PartDesign::", "Sketcher::", "Spreadsheet::"))])
{"sheet.set('B1', '=%s mm'); doc.recompute(); out['resized_volume']=body.Shape.Volume; out['resized_z']=body.Shape.BoundBox.ZLength; out['bad']=[o.Name for o in doc.Objects if not o.isValid()]" % resize_to if resize_to else ""}
print("RESULT " + json.dumps(out))
''')
    p = subprocess.run([FREECADCMD, str(script)], capture_output=True, text=True, timeout=120)
    lines = [l for l in (p.stdout + p.stderr).splitlines() if "RESULT " in l]
    assert lines, p.stdout[-800:] + p.stderr[-800:]
    return json.loads(lines[-1].split("RESULT ", 1)[1])


def test_macro_builds_a_valid_fully_constrained_parametric_sphere(tmp_path):
    r = run_macro(tmp_path, 60.0)
    assert r["valid"] and r["solve"] == 0 and r["fully"] is True
    assert r["volume"] == pytest.approx(4 / 3 * math.pi * 30 ** 3, rel=1e-4)
    assert r["zlen"] == pytest.approx(60) and r["zmin"] == pytest.approx(0, abs=1e-6)  # sits on the ground
    assert ["PartDesign::Body"] != r["tree"] and "PartDesign::Revolution" in r["tree"] and "Sketcher::SketchObject" in r["tree"]
    assert ["Constraints.Diameter", "Params.Diameter"] in [[e[0].lstrip("."), e[1]] for e in r["expr"]]
    assert "PLACEHOLDER" not in r["label"]  # SCALE_CONFIRMED = True


def test_editing_one_spreadsheet_cell_regenerates_the_model(tmp_path):
    r = run_macro(tmp_path, 60.0, resize_to=100)
    assert r["resized_volume"] == pytest.approx(4 / 3 * math.pi * 50 ** 3, rel=1e-4)
    assert r["resized_z"] == pytest.approx(100) and r["bad"] == []


def test_placeholder_size_is_labelled_until_confirmed(tmp_path):
    src = set_var(set_var(MACRO.read_text(), "DIAMETER_MM", 50.0), "SCALE_CONFIRMED", False)
    (tmp_path / "r.py").write_text(src + "\nprint('RESULT ' + repr(App.ActiveDocument.Label))\n")
    p = subprocess.run([FREECADCMD, str(tmp_path / "r.py")], capture_output=True, text=True, timeout=120)
    assert "PLACEHOLDER SIZE" in p.stdout + p.stderr


def test_exported_stl_round_trips_through_our_measure_tool(tmp_path):
    """FreeCAD sphere -> STL -> measure/: the numbers must come back as the diameter we put in."""
    pytest.importorskip("open3d")
    from measure import analyze
    stl = tmp_path / "ball.stl"
    run_macro(tmp_path, 80.0, export=stl)
    assert stl.exists() and stl.stat().st_size > 1000
    r = analyze(stl, 1.0, plane_removal=False)  # STL from FreeCAD is in mm, so scale 1
    sph = r["primitive_fit"]["sphere"]
    assert sph["diameter_mm"] == pytest.approx(80, rel=0.01)
    assert r["primitive_fit"]["likely_shape"] == "sphere" and r["primitive_fit"]["fit_quality"] == "good"
    assert r["volume"]["mm3"] == pytest.approx(4 / 3 * math.pi * 40 ** 3, rel=0.02)


def test_shipped_defaults_are_the_measured_44mm_ball(tmp_path):
    """The macro as shipped (no overrides) builds the real 44 mm ball, with the size marked as measured."""
    (tmp_path / "r.py").write_text(MACRO.read_text() + """
import json
print('RESULT ' + json.dumps(dict(label=App.ActiveDocument.Label, vol=App.ActiveDocument.Body.Shape.Volume,
      z=App.ActiveDocument.Body.Shape.BoundBox.ZLength)))
""")
    p = subprocess.run([FREECADCMD, str(tmp_path / "r.py")], capture_output=True, text=True, timeout=120)
    r = json.loads([l for l in (p.stdout + p.stderr).splitlines() if "RESULT " in l][-1].split("RESULT ", 1)[1])
    assert r["z"] == pytest.approx(44.0) and r["vol"] == pytest.approx(4 / 3 * math.pi * 22 ** 3, rel=1e-4)
    assert "PLACEHOLDER" not in r["label"] and "PLACEHOLDER" not in p.stdout + p.stderr
