#!/usr/bin/env python3
"""
scan2cad.py  --  photos/video of an object  ->  parametric FreeCAD model (+ STEP for Fusion 360)

Pipeline
  1. Pick sharp, evenly spread keyframes from a video or a folder of photos
  2. Detect an ArUco marker of known size in each frame for real-world scale
  3. Ask an OpenAI vision model to write a parametric FreeCAD Python script
  4. Run the script headless with freecadcmd; on error, send the traceback back and retry
  5. Render the result, show the model photos vs. renders, let it correct itself
  6. Save model.FCStd, model.step (import into Fusion), model.stl, params.json, script

Setup (once)
  pip install openai opencv-contrib-python numpy trimesh matplotlib
  Mac/Linux:  export OPENAI_API_KEY="sk-..."
  Windows:    setx OPENAI_API_KEY "sk-..."   (then open a new terminal)

Usage
  python scan2cad.py --test-key --quality all   # check access to both preset models (two tiny calls)
  python scan2cad.py --selftest-freecad         # check FreeCAD + rendering (no API cost)
  python scan2cad.py --make-marker              # writes marker_0_front.png ... marker_5_back_left_corner.png
  python scan2cad.py --input scan.mp4 --marker-mm 40 --quality medium
  python scan2cad.py --input photos/ --marker-mm 40 --quality high --describe "3D-printed bracket with 2 holes"
  python scan2cad.py --run-code my_script.py --name "Bracket"  # run/render without API calls
"""
import argparse
import base64
import glob
import json
import os
import platform
import re
import shutil
import site
import subprocess
import sys
import time
from pathlib import Path

# Prefer dependencies installed beside this script, then make the normal user
# site visible as well. FreeCAD's bundled Windows Python omits user site-packages
# from sys.path even when they are enabled.
local_packages = Path(__file__).resolve().parent / ".packages"
if local_packages.is_dir():
    sys.path.insert(0, str(local_packages))
user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    site.addsitedir(user_site)

import cv2
import numpy as np

QUALITY_PRESETS = {
    "low":    {"model": "gpt-6-sol",   "effort": "low",  "frames": 12},
    "medium": {"model": "gpt-6-sol",   "effort": "high", "frames": 16},
    "high":   {"model": "gpt-6-astra", "effort": "low",  "frames": 24},
    "max":    {"model": "gpt-6-astra", "effort": "high", "frames": 32},
}
MAX_SIDE = 1024                      # images are downscaled to this before sending
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_LAYOUT = {
    0: "front",
    1: "right",
    2: "back",
    3: "left",
    4: "back-right corner",
    5: "back-left corner",
}

# --------------------------------------------------------------------------- prompts

INSTRUCTIONS = """You are an expert mechanical CAD engineer. You reconstruct real objects as clean,
parametric FreeCAD models from photos.

Write a FreeCAD Python script that follows this contract exactly:
- It runs headless in freecadcmd. Import FreeCAD, Part (and math if needed). Never import FreeCADGui or any GUI module.
- It starts with a PARAMETERS block: every dimension is an UPPER_CASE variable in millimetres on its own line
  with a short comment ending in a confidence tag [high], [med], or [low],
  e.g.  BODY_WIDTH = 42.0  # overall width [high]
- After the PARAMETERS block, include a "# ASSUMPTIONS" comment block listing every guess about surfaces not
  visible in any photo (for example, the bottom). Assume the simplest plausible geometry for unseen surfaces.
- All geometry is derived from those parameters (no unexplained magic numbers), so editing a parameter
  and re-running regenerates the model correctly.
- DESIGN INTENT: snap estimated dimensions to plausible designed values: whole millimetres, or 0.5 mm increments
  for small features. Holes are usually standard clearance sizes: M2 2.4 mm, M3 3.4 mm, M4 4.5 mm,
  M5 5.5 mm, and M6 6.6 mm. 3D-printed walls are usually 1.2-3 mm. Assume symmetry when the object looks symmetric.
- Structure the geometry as named features in manufacturing order: base body, added features, removed features
  (holes, pockets, and slots), then fillets/chamfers. Put a one-line comment above each feature naming it.
- Build the solid from Part primitives and boolean operations (Part.makeBox, makeCylinder, makeCone,
  makeSphere, makeTorus, extruded wires/faces, fuse, cut, common, makeFillet, makeChamfer).
- Also define COLORS as a dict mapping part names to hex colors (for example, "#C84A32") and PARTS as a
  dict mapping the same names to the Part sub-solids carrying those colors. Estimate the object's true surface
  colors: correct for lighting and white balance by treating the ArUco marker's white paper as pure white and
  its black areas as pure black, and ignore highlights and reflections. For a single-color object, use one entry.
- Assign the final single solid to a variable named `result`.
- `result` must remain the single fused solid even when COLORS and PARTS are defined.
- Do not create documents, save, or export files. The harness does that.
- Coordinate system: Z is up, the object rests on the XY plane (z = 0) and is centred on the origin in X and Y.
  "Front" is the side seen when looking from -Y toward +Y.
- Use the table's marker layout to orient the model consistently: ID 0 is at the object's front and defines the
  -Y side of the CAD coordinate system; ID 1 is at the object's right and defines +X; ID 2 is at the back;
  ID 3 is at the left; IDs 4 and 5 are at the back-right and back-left corners. Orient visible object features
  to match these directions rather than choosing an arbitrary rotation.

Model the features you can actually see (holes, slots, steps, bosses, pockets, fillets, chamfers).
Prefer a slightly simpler model that is dimensionally right over a detailed one that is wrong.
Ignore the table, the scale marker, the robot, and background clutter.
Return the complete script in one ```python code block."""

FIRST_PROMPT = """Reconstruct the object in these photos as a parametric FreeCAD model.

Scale reference: the red-outlined square is an ArUco marker exactly {marker} mm wide, lying near the object.
Per-image scale where the marker was detected:
{scale_notes}
Marker layout around the object: ID 0 front (-Y), ID 1 right (+X), ID 2 back (+Y), ID 3 left (-X),
ID 4 back-right corner, ID 5 back-left corner. Use the per-photo camera-direction estimates and this layout
to orient the model so its front and right sides agree with the CAD coordinate system.
The px/mm value only holds at the marker's distance from the camera; parts of the object nearer the camera
look larger. Cross-check dimensions between views.
{describe}
Before the script, provide a table with exactly these columns:
parameter | estimated mm | which photo(s) | how measured
For each measurable value, cite the visual measurement explicitly (for example, "2.1 marker widths in Photo 3").
Mark every unmeasurable value as GUESS.
Then briefly list the object's features and your estimated dimensions, with how you derived each one.
Immediately before the script, write exactly one line in this form:
NAME: <2-4 word title>
Name the object concisely and omit filler words (for example, "yellow roll of electrical tape" becomes
"Yellow Electrical Tape").
Then give the complete script."""

FIX_PROMPT = """This FreeCAD script failed when run in freecadcmd.

```python
{code}
```

Error:
```
{err}
```

Fix it while keeping the same parameters and design intent. Return the complete corrected script in one ```python block."""

REFINE_PROMPT = """Above are the original photos, followed by renders of your current model: orthographic views
with mm axes, and the title gives the model's bounding box. {scale_hint}
{warn}
Current script:
```python
{code}
```

Compare carefully: overall size and proportions (use the marker scale), missing or extra features,
hole and slot positions and diameters, orientation, and fillets/chamfers. List each discrepancy separately,
naming the affected parameter and its corrected value. Fix only what is clearly wrong; do not redesign parts
that already match the photos.
Check each modeled surface color against the photos, correcting for white balance and ignoring highlights and reflections.
If the model already matches the object well, reply with exactly MATCH on the first line and nothing else.
Otherwise list the discrepancies, then return the complete corrected script in one ```python block."""

# FreeCAD-side harness: runs the model's code, then saves/exports. Runs inside freecadcmd.
RUNNER = r'''
import os, json, re, traceback
import FreeCAD, Part
out = os.environ["S2C_OUT"]
code_path = os.environ["S2C_CODE"]
try:
    ns = {"__name__": "__s2c__"}
    with open(code_path, encoding="utf-8") as f:
        exec(compile(f.read(), code_path, "exec"), ns)
    shape = ns.get("result")
    if shape is None:
        raise RuntimeError("The script did not assign the final solid to a variable named `result`.")
    if not isinstance(shape, Part.Shape) and hasattr(shape, "Shape"):
        shape = shape.Shape
    if shape.isNull():
        raise RuntimeError("`result` is a null (empty) shape.")
    doc = FreeCAD.newDocument("scan2cad")
    obj = doc.addObject("Part::Feature", "Scanned")
    obj.Shape = shape
    doc.recompute()
    doc.saveAs(os.path.join(out, "model.FCStd"))
    shape.exportStep(os.path.join(out, "model.step"))
    shape.exportStl(os.path.join(out, "model.stl"))
    colors = ns.get("COLORS")
    parts = ns.get("PARTS")
    if isinstance(colors, dict) and isinstance(parts, dict):
        if set(colors) != set(parts):
            raise RuntimeError("COLORS and PARTS must contain exactly the same names.")
        clean_colors = {}
        used_filenames = set()
        for name, color in colors.items():
            if not isinstance(name, str) or not isinstance(color, str):
                raise RuntimeError("COLORS must map string names to hex color strings.")
            if not re.fullmatch(r"#[0-9A-Fa-f]{6}", color):
                raise RuntimeError("Color for %r must use #RRGGBB format." % name)
            filename_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "part"
            filename = "part_%s.stl" % filename_name
            if filename.lower() in used_filenames:
                raise RuntimeError("PARTS names produce duplicate filenames: %r" % name)
            used_filenames.add(filename.lower())
            part = parts[name]
            if not isinstance(part, Part.Shape) and hasattr(part, "Shape"):
                part = part.Shape
            if not isinstance(part, Part.Shape) or part.isNull():
                raise RuntimeError("PARTS[%r] is not a non-empty Part shape." % name)
            part.exportStl(os.path.join(out, filename))
            part.exportStep(os.path.join(out, "part_%s.step" % filename_name))
            clean_colors[name] = color.upper()
        with open(os.path.join(out, "colors.json"), "w", encoding="utf-8") as f:
            json.dump(clean_colors, f, indent=2)
    bb = shape.BoundBox
    info = dict(xlen=bb.XLength, ylen=bb.YLength, zlen=bb.ZLength, volume=shape.Volume,
                valid=shape.isValid(), solids=len(shape.Solids))
    with open(os.path.join(out, "info.json"), "w") as f:
        json.dump(info, f)
except Exception:
    with open(os.path.join(out, "error.txt"), "w", encoding="utf-8") as f:
        f.write(traceback.format_exc())
'''

SELFTEST_CODE = '''import FreeCAD, Part
# PARAMETERS
LENGTH = 60.0  # block length (X)
WIDTH = 30.0  # block width (Y)
HEIGHT = 15.0  # block height (Z)
HOLE_DIA = 8.0  # through-hole diameter
HOLE_SPACING = 36.0  # centre-to-centre distance between holes

body = Part.makeBox(LENGTH, WIDTH, HEIGHT, FreeCAD.Vector(-LENGTH / 2, -WIDTH / 2, 0))
for x in (-HOLE_SPACING / 2, HOLE_SPACING / 2):
    body = body.cut(Part.makeCylinder(HOLE_DIA / 2, HEIGHT, FreeCAD.Vector(x, 0, 0)))
result = body
'''

# --------------------------------------------------------------------------- images


def sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def downscale(img):
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def load_frames(inp, video_step):
    p = Path(inp)
    frames = []
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                img = cv2.imread(str(f))
                if img is not None:
                    frames.append((f.stem, img))
    else:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            sys.exit(f"Could not open {p}")
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i % video_step == 0:
                frames.append((f"frame{i:05d}", img))
            i += 1
        cap.release()
    if not frames:
        sys.exit(f"No images found in {inp}")
    return frames


def pick_keyframes(source, k, video_step=5, marker_mm=60, selection_out=None, groups=12):
    """Drop blurry frames, then take the sharpest frame from each of k segments (in capture order),
    which gives even coverage around a video orbit. PNG folders use select_frames' richer pipeline."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir() and any(child.is_file() and child.suffix.lower() in {".png", ".jpg", ".jpeg"}
                                 for child in path.iterdir()):
            import select_frames
            selected, _ = select_frames.select_frames(
                path, n=k, out_folder=selection_out, marker_mm=marker_mm,
                groups=groups, preview=False)
            frames = []
            for selected_path in selected:
                img = cv2.imread(str(selected_path))
                if img is not None:
                    frames.append((selected_path.stem, img))
            if not frames:
                sys.exit(f"Frame selector returned no readable images from {path}")
            return frames
        frames = load_frames(source, video_step)
    else:
        frames = source
    scores = [sharpness(img) for _, img in frames]
    med = float(np.median(scores))
    keep = [(n, img, s) for (n, img), s in zip(frames, scores) if s >= 0.35 * med]
    if len(keep) <= k:
        return [(n, img) for n, img, _ in keep]
    chosen = []
    for seg in np.array_split(np.arange(len(keep)), k):
        j = max(seg, key=lambda i: keep[i][2])
        chosen.append((keep[j][0], keep[j][1]))
    return chosen


_detector = None


def estimate_view_direction(markers, image_shape):
    """Estimate camera side from known table positions and perspective cues."""
    if not markers:
        return "camera direction unknown (no layout markers visible)"

    h, w = image_shape[:2]
    layout_vectors = {
        0: (0.0, -1.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (-1.0, 0.0),
        4: (0.7, 0.7), 5: (-0.7, 0.7),
    }
    median_side = float(np.median([m["side_px"] for m in markers]))
    vx = vy = total = 0.0
    observations = []
    for marker in markers:
        cx, cy = marker["center"]
        x_pos = "left" if cx < w / 3 else "right" if cx > 2 * w / 3 else "center"
        if len(markers) == 1:
            depth = "only marker"
        elif marker["side_px"] > median_side * 1.15 or cy > h * 0.67:
            depth = "near/larger"
        elif marker["side_px"] < median_side * 0.85 or cy < h * 0.33:
            depth = "far/smaller"
        else:
            depth = "mid-distance"
        observations.append(f"ID {marker['id']} {MARKER_LAYOUT[marker['id']]} at image {x_pos}, {depth}")

        size_weight = marker["side_px"] / max(median_side, 1e-6)
        vertical_weight = 0.75 + 0.5 * (cy / max(h, 1))
        center_weight = 1.0 + 0.1 * (1.0 - min(abs(cx / max(w, 1) - 0.5) * 2.0, 1.0))
        weight = size_weight * vertical_weight * center_weight
        dx, dy = layout_vectors[marker["id"]]
        vx += dx * weight
        vy += dy * weight
        total += weight
    vx, vy = vx / total, vy / total
    if abs(vx) < 0.2 and abs(vy) < 0.2:
        closest = max(markers, key=lambda m: (m["side_px"], m["center"][1]))
        vx, vy = layout_vectors[closest["id"]]

    vertical = "front" if vy < -0.2 else "back" if vy > 0.2 else ""
    horizontal = "right" if vx > 0.2 else "left" if vx < -0.2 else ""
    camera_side = "-".join(x for x in (vertical, horizontal) if x) or MARKER_LAYOUT[markers[0]["id"]]
    faces = " and ".join(x for x in (vertical, horizontal) if x)
    if not faces:
        faces = camera_side.replace("-", " and ")
    face_word = "face" if " and " not in faces else "faces"
    return (f"camera at {camera_side}, looking at the {faces} {face_word}; "
            + "; ".join(observations))


def detect_marker(img, marker_mm):
    """Return annotated image, median px/mm, accepted marker details, view note, and scale warning."""
    global _detector
    if _detector is None:
        _detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
                                            cv2.aruco.DetectorParameters())
    corners, ids, _ = _detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    out = img.copy()
    markers = []
    if ids is not None:
        for marker_corners, marker_id_raw in zip(corners, ids.flatten()):
            marker_id = int(marker_id_raw)
            if marker_id not in MARKER_LAYOUT:
                continue
            c = marker_corners.reshape(4, 2)
            side_px = float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))
            center = tuple(np.mean(c, axis=0).tolist())
            ppm = side_px / marker_mm if marker_mm else None
            markers.append({"id": marker_id, "side_px": side_px, "center": center, "px_per_mm": ppm})
            cv2.polylines(out, [c.astype(np.int32)], True, (0, 0, 255), 2)
            x, y = c[0].astype(int)
            label = f"ID {marker_id} {MARKER_LAYOUT[marker_id]}"
            if marker_mm:
                label += f" {marker_mm:g} mm"
            cv2.putText(out, label, (int(x), int(max(y - 8, 14))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    ppm_values = [m["px_per_mm"] for m in markers if m["px_per_mm"] is not None]
    median_ppm = float(np.median(ppm_values)) if ppm_values else None
    scale_warning = ""
    if len(ppm_values) > 1 and (max(ppm_values) - min(ppm_values)) / median_ppm > 0.25:
        scale_warning = "WARNING: marker scales disagree by more than 25%; markers are likely at different distances"
    return out, median_ppm, markers, estimate_view_direction(markers, img.shape), scale_warning


def to_data_url(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

# --------------------------------------------------------------------------- model API


def ask(client, model, effort, text, images=(), log=None, tag=""):
    content = []
    for label, img in images:
        content.append({"type": "input_text", "text": label})
        content.append({"type": "input_image", "image_url": to_data_url(img)})
    content.append({"type": "input_text", "text": text})
    t0 = time.time()
    kwargs = dict(model=model, instructions=INSTRUCTIONS, input=[{"role": "user", "content": content}])
    if effort:
        kwargs["reasoning"] = {"effort": effort}
    resp = client.responses.create(**kwargs)
    out = resp.output_text or ""
    u = getattr(resp, "usage", None)
    print(f"      model replied in {time.time() - t0:.0f}s "
          f"(tokens in={getattr(u, 'input_tokens', '?')}, out={getattr(u, 'output_tokens', '?')})")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n\n## {tag}\n\n### prompt\n{text}\n\n### reply\n{out}\n")
    return out


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", text, re.S)
    return blocks[-1].strip() + "\n" if blocks else None


def resolve_quality(quality="medium", model=None, effort=None, frames=None, groups=None):
    """Resolve a quality preset plus hidden developer overrides."""
    if quality not in QUALITY_PRESETS:
        raise ValueError(f"quality must be one of: {', '.join(QUALITY_PRESETS)}")
    preset = QUALITY_PRESETS[quality]
    resolved_frames = frames if frames is not None else preset["frames"]
    return {
        "model": model or preset["model"],
        "effort": effort or preset["effort"],
        "frames": resolved_frames,
        "groups": groups if groups is not None else max(1, resolved_frames * 3 // 4),
    }


def extract_name(text):
    """Extract the concise object title requested from the initial model reply."""
    match = re.search(r"(?im)^\s*(?:\*\*)?NAME(?:\*\*)?\s*:\s*(.+?)\s*$", text or "")
    if not match:
        return None
    return match.group(1).strip().strip("`*_# ") or None


def fallback_name_from_description(description):
    """Turn a free-form description into a useful short title."""
    filler = {"a", "an", "the", "of", "this", "that", "object", "item", "thing",
              "photo", "photos", "picture", "scan", "scanned", "roll"}
    words = re.findall(r"[A-Za-z0-9]+", description or "")
    return " ".join(word for word in words if word.lower() not in filler) or None


def safe_model_stem(name):
    """Return a Windows-safe Title_Case_With_Underscores filename stem."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", name or "")
    words = re.findall(r"[A-Za-z0-9]+", cleaned)
    stem = "_".join(word[:1].upper() + word[1:].lower() for word in words)
    stem = stem[:60].rstrip(" ._")
    stem = stem or "Scan"
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}
    return f"Scan_{stem}" if stem.upper() in reserved else stem


def unique_step_path(models_dir, stem):
    """Choose stem.STEP, stem_2.STEP, ... without overwriting an existing export."""
    candidate = models_dir / f"{stem}.STEP"
    suffix = 2
    while candidate.exists():
        suffix_text = f"_{suffix}"
        candidate = models_dir / f"{stem[:60 - len(suffix_text)].rstrip(' ._')}{suffix_text}.STEP"
        suffix += 1
    return candidate


def save_named_step(final, args, model_name=None):
    """Copy only the best available STEP into the user's named-model collection."""
    models_dir = Path(args.models_dir) if args.models_dir else Path(__file__).resolve().parent.parent / "Scanned_Models"
    models_dir.mkdir(parents=True, exist_ok=True)
    raw_name = (args.name or model_name or fallback_name_from_description(args.describe)
                or f"Scan {time.strftime('%Y%m%d-%H%M%S')}")
    destination = unique_step_path(models_dir, safe_model_stem(raw_name))
    colored = final / "model_color.step"
    source = colored if colored.exists() else final / "model.step"
    if not colored.exists():
        print("WARNING: colored STEP was not produced; the named saved file has no color.")
    shutil.copy2(source, destination)
    print(f"  Named STEP saved to: {destination.resolve()}")
    return destination


def extract_params(code):
    params = {}
    pat = r"^([A-Z][A-Z0-9_]*)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(?:#\s*(.*))?$"
    for m in re.finditer(pat, code, re.M):
        params[m.group(1)] = {"value_mm": float(m.group(2)), "note": (m.group(3) or "").strip()}
    return params

# --------------------------------------------------------------------------- FreeCAD


def find_freecadcmd(override):
    if override:
        return override
    for name in ("freecadcmd", "FreeCADCmd", "freecad.cmd"):
        p = shutil.which(name)
        if p:
            return p
    cands = []
    system = platform.system()
    if system == "Darwin":
        cands = ["/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd",
                 "/Applications/FreeCAD.app/Contents/Resources/bin/FreeCADCmd",
                 "/Applications/FreeCAD.app/Contents/MacOS/FreeCADCmd"]
    elif system == "Windows":
        roots = [os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")]
        for root in roots:
            cands += sorted(glob.glob(os.path.join(root, "FreeCAD*", "bin", "freecadcmd.exe")), reverse=True)
            cands += sorted(glob.glob(os.path.join(root, "FreeCAD*", "bin", "FreeCADCmd.exe")), reverse=True)
    else:
        cands = ["/usr/bin/freecadcmd", "/usr/bin/FreeCADCmd", "/usr/local/bin/freecadcmd", "/snap/bin/freecad.cmd"]
    for c in cands:
        if os.path.exists(c):
            return c
    sys.exit("Couldn't find freecadcmd. Pass its location with --freecadcmd /path/to/freecadcmd")


def run_freecad(fc, code, work):
    work.mkdir(parents=True, exist_ok=True)
    for f in ("error.txt", "info.json", "model.stl", "model.step", "model.FCStd",
              "colors.json", "model_color.glb", "model_color.3mf", "model_color.step"):
        (work / f).unlink(missing_ok=True)
    for f in work.glob("part_*.stl"):
        f.unlink()
    for f in work.glob("part_*.step"):
        f.unlink()
    code_path = work / "model_code.py"
    code_path.write_text(code, encoding="utf-8")
    runner = work / "_runner.py"
    runner.write_text(RUNNER, encoding="utf-8")
    env = dict(os.environ, S2C_OUT=str(work.resolve()), S2C_CODE=str(code_path.resolve()))
    try:
        p = subprocess.run([fc, str(runner.resolve())], env=env, capture_output=True, text=True,
                           timeout=240, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None, "FreeCAD timed out after 240 s (geometry too complex, or an endless loop)."
    if (work / "error.txt").exists():
        return None, (work / "error.txt").read_text(encoding="utf-8")
    if not (work / "info.json").exists():
        return None, ("FreeCAD produced no output.\nstdout:\n" + (p.stdout or "")[-3000:] +
                      "\nstderr:\n" + (p.stderr or "")[-3000:])
    return json.loads((work / "info.json").read_text()), None


def colored_meshes(work, stl_path):
    """Load colored part meshes when present, otherwise return the fused mesh in default blue."""
    import trimesh

    colors_path = work / "colors.json"
    if not colors_path.exists():
        return [(trimesh.load(str(stl_path), force="mesh"), np.array([0.55, 0.68, 0.85]))]
    color_data = json.loads(colors_path.read_text(encoding="utf-8"))
    meshes = []
    for name, hex_color in color_data.items():
        mesh = trimesh.load(str(work / part_stl_name(name)), force="mesh")
        rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], dtype=float) / 255.0
        meshes.append((mesh, rgb))
    return meshes


def part_stl_name(name):
    """Return the same safe part filename used by the FreeCAD runner."""
    filename_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "part"
    return f"part_{filename_name}.stl"


def part_step_name(name):
    """Return the STEP equivalent of the safe part filename used by the FreeCAD runner."""
    return str(Path(part_stl_name(name)).with_suffix(".step"))


def export_colored_step(work, color_data):
    """Create a color-preserving STEP assembly with CadQuery when it is available."""
    try:
        import cadquery as cq

        assy = cq.Assembly()
        for name, hex_color in color_data.items():
            part = cq.importers.importStep(str(work / part_step_name(name)))
            rgb = [int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
            assy.add(part, name=name, color=cq.Color(*rgb))
        output = str(work / "model_color.step")
        if callable(getattr(assy, "save", None)):
            assy.save(output)
        elif callable(getattr(assy, "export", None)):
            assy.export(output)
        else:
            raise RuntimeError("this CadQuery version has neither Assembly.save nor Assembly.export")
    except Exception as exc:
        print(f"WARNING: colored STEP export skipped: {type(exc).__name__}: {exc}")


def export_colored_models(work):
    """Combine colored part STLs into portable GLB and 3MF scene files."""
    if not (work / "colors.json").exists():
        return
    import trimesh

    scene = trimesh.Scene()
    color_data = json.loads((work / "colors.json").read_text(encoding="utf-8"))
    for name, hex_color in color_data.items():
        mesh = trimesh.load(str(work / part_stl_name(name)), force="mesh")
        rgba = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)] + [255]
        mesh.visual.face_colors = rgba
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    scene.export(str(work / "model_color.glb"))
    scene.export(str(work / "model_color.3mf"))
    export_colored_step(work, color_data)


def render(stl_path, png_path, info):
    """Four orthographic views, using colored part meshes when available."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    meshes = colored_meshes(stl_path.parent, stl_path)
    bounds = np.array([mesh.bounds for mesh, _ in meshes])
    lo, hi = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    mid, r = (lo + hi) / 2, max((hi - lo).max() / 2, 1e-3) * 1.05
    views = [("Front (from -Y)", 0, -90), ("Right (from +X)", 0, 0), ("Top (from +Z)", 90, -90), ("Isometric", 25, -55)]
    fig = plt.figure(figsize=(10, 10))
    for i, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d", proj_type="ortho")
        ax.view_init(elev=elev, azim=azim)
        e, a = np.radians(elev), np.radians(azim)
        cam = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        for mesh, rgb in meshes:
            tris = mesh.vertices[mesh.faces]
            shade = 0.25 + 0.7 * np.clip(mesh.face_normals @ cam, 0, 1)
            facecolors = np.c_[shade[:, None] * rgb[None, :], np.ones_like(shade)]
            ax.add_collection3d(Poly3DCollection(tris, facecolors=facecolors, edgecolor="none"))
        ax.set_xlim(mid[0] - r, mid[0] + r)
        ax.set_ylim(mid[1] - r, mid[1] + r)
        ax.set_zlim(mid[2] - r, mid[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("X mm")
        ax.set_ylabel("Y mm")
        ax.set_zlabel("Z mm")
        ax.set_title(title)
    fig.suptitle(f"Model bounding box: {info['xlen']:.1f} x {info['ylen']:.1f} x {info['zlen']:.1f} mm  (X x Y x Z)")
    fig.tight_layout()
    fig.savefig(str(png_path), dpi=90)
    plt.close(fig)
    return cv2.imread(str(png_path))

# --------------------------------------------------------------------------- main steps


def build_until_it_runs(client, args, fc, code, work, log):
    """Run the script; on failure send the traceback back for a fix. Returns (code, info or None)."""
    for attempt in range(args.max_fix + 1):
        info, err = run_freecad(fc, code, work)
        if info:
            return code, info
        last_line = err.strip().splitlines()[-1] if err.strip() else "unknown error"
        print(f"      FreeCAD error: {last_line}")
        if attempt == args.max_fix or client is None:
            break
        print(f"      asking the model to fix it (attempt {attempt + 1}/{args.max_fix})...")
        reply = ask(client, args.model, "low", FIX_PROMPT.format(code=code, err=err[-4000:]),
                    log=log, tag=f"{work.name} fix {attempt + 1}")
        code = extract_code(reply) or code
    return code, None


def finish(out, work, code, info, args, model_name=None):
    final = out / "final"
    final.mkdir(parents=True, exist_ok=True)
    for f in ("model_color.glb", "model_color.3mf", "model_color.step", "colors.json"):
        if not (work / f).exists():
            (final / f).unlink(missing_ok=True)
    for f in ("model.FCStd", "model.step", "model.stl", "render.png",
              "model_color.glb", "model_color.3mf", "model_color.step", "colors.json"):
        if (work / f).exists():
            shutil.copy(work / f, final / f)
    (final / "model_code.py").write_text(code, encoding="utf-8")
    params = extract_params(code)
    (final / "params.json").write_text(json.dumps({"bounding_box_mm": info, "parameters": params}, indent=2))
    print(f"\nDone. Final model in {final}")
    print(f"  bounding box: {info['xlen']:.1f} x {info['ylen']:.1f} x {info['zlen']:.1f} mm, "
          f"valid solid: {info['valid']}")
    for k, v in params.items():
        print(f"  {k:<24} {v['value_mm']:>9.2f} mm   {v['note']}")
    print("\n  FreeCAD: open model.FCStd, or edit the PARAMETERS in model_code.py and re-run with --run-code")
    print("  Fusion 360: File > Open (or Insert) > model.step")
    save_named_step(final, args, model_name)


def cmd_test_key(args):
    from openai import OpenAI
    models = ["gpt-6-sol", "gpt-6-astra"] if args.quality == "all" else [args.model]
    failed = False
    for model in models:
        try:
            r = OpenAI().responses.create(model=model, input="Reply with just the word OK.")
            print(f"Key works and {model} is reachable. Reply: {r.output_text!r}")
        except Exception as e:
            failed = True
            print(f"API call failed for model {model}:\n  {type(e).__name__}: {e}")
    if failed:
        print("Check: key set in this terminal? credit added at platform.openai.com? model name right?")


def cmd_make_marker():
    """One marker per MARKER_LAYOUT position (IDs 0-5): scan2cad only reads the ID it sees in
    a photo to place that photo at front/right/back/left/a corner, so a single reused marker
    can't tell those positions apart -- each position needs its own distinct ID printed."""
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    for marker_id, position in MARKER_LAYOUT.items():
        m = cv2.aruco.generateImageMarker(d, marker_id, 600)
        m = cv2.copyMakeBorder(m, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
        slug = re.sub(r"[^a-z0-9]+", "_", position.lower()).strip("_")
        filename = f"marker_{marker_id}_{slug}.png"
        cv2.imwrite(filename, m)
        print(f"Wrote {filename}  (ID {marker_id}, {position})")
    print("\nPrint all six at 100% scale (no 'fit to page') -- they must all come out the same physical size.")
    print("Measure one BLACK square's width in mm with calipers or a ruler, and pass that number as --marker-mm.")
    print("Tape each one flat at its labeled position around the object. Leaving a position uncovered is fine;")
    print("scan2cad just falls back to a guess for dimensions/orientation on sides with no marker in view.")


def cmd_run_code(args, code, label):
    fc = find_freecadcmd(args.freecadcmd)
    out = Path(args.out or f"runs/{label}")
    work = out / "round0"
    print(f"Running {label} with {fc} ...")
    info, err = run_freecad(fc, code, work)
    if not info:
        sys.exit(f"FreeCAD failed:\n{err}")
    export_colored_models(work)
    render(work / "model.stl", work / "render.png", info)
    finish(out, work, code, info, args)


def main():
    ap = argparse.ArgumentParser(description="Photos/video of an object -> parametric FreeCAD model")
    ap.add_argument("--input", help="video file or folder of photos")
    ap.add_argument("--marker-mm", type=float, help="measured width of the printed ArUco marker's black square")
    ap.add_argument("--describe", default="", help='optional hint, e.g. "PLA bracket with two M4 holes"')
    ap.add_argument("--name", help="name for the final STEP export (highest priority)")
    ap.add_argument("--models-dir", help="folder for the named final STEP (default: ../Scanned_Models)")
    ap.add_argument("--out", help="output folder (default: runs/<timestamp>)")
    ap.add_argument("--quality", choices=[*QUALITY_PRESETS, "all"], default="medium",
                    help="quality preset: low, medium, high, or max (default medium); 'all' is for --test-key")
    ap.add_argument("--frames", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--groups", type=int, default=None,
                    help="equal-motion frame groups (default: 3/4 of selected frames)")
    ap.add_argument("--video-step", type=int, default=5, help="sample every Nth video frame (default 5)")
    ap.add_argument("--rounds", type=int, default=2, help="compare-and-correct rounds (default 2)")
    ap.add_argument("--max-fix", type=int, default=3, help="retries when a script errors (default 3)")
    ap.add_argument("--model", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--effort", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--freecadcmd", help="path to freecadcmd if not found automatically")
    ap.add_argument("--test-key", action="store_true")
    ap.add_argument("--make-marker", action="store_true")
    ap.add_argument("--selftest-freecad", action="store_true")
    ap.add_argument("--run-code", help="run and render an existing FreeCAD script, no API calls")
    args = ap.parse_args()

    if args.quality == "all":
        if not args.test_key:
            ap.error("--quality all is only valid with --test-key")
        medium = resolve_quality("medium", args.model, args.effort, args.frames, args.groups)
        for key, value in medium.items():
            setattr(args, key, value)
        print("Quality: all -> testing gpt-6-sol and gpt-6-astra")
    else:
        settings = resolve_quality(args.quality, args.model, args.effort, args.frames, args.groups)
        for key, value in settings.items():
            setattr(args, key, value)
        print(f"Quality: {args.quality} -> {args.model}, effort {args.effort}, {args.frames} frames")

    if args.make_marker:
        return cmd_make_marker()
    if args.selftest_freecad:
        return cmd_run_code(args, SELFTEST_CODE, "selftest")
    if args.run_code:
        return cmd_run_code(args, Path(args.run_code).read_text(encoding="utf-8"), Path(args.run_code).stem)
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set in this terminal. See the setup notes at the top of this file.")
    if args.test_key:
        return cmd_test_key(args)
    if not args.input:
        ap.error("--input is required (a video file or a folder of photos)")

    from openai import OpenAI
    client = OpenAI()
    fc = find_freecadcmd(args.freecadcmd)
    out = Path(args.out or time.strftime("runs/%Y%m%d-%H%M%S"))
    (out / "keyframes").mkdir(parents=True, exist_ok=True)
    log = out / "log.md"

    # 1-2. keyframes + scale
    print("1. Selecting keyframes...")
    frames = pick_keyframes(args.input, args.frames, args.video_step, args.marker_mm,
                            out / "frame_selection", args.groups)
    photos, notes = [], []
    marker_photo_counts = {marker_id: 0 for marker_id in MARKER_LAYOUT}
    for i, (name, img) in enumerate(frames, 1):
        img, ppm, markers, view_note, scale_warning = detect_marker(downscale(img), args.marker_mm)
        cv2.imwrite(str(out / "keyframes" / f"{i:02d}_{name}.jpg"), img)
        photos.append((f"Photo {i}:", img))
        for marker_id in {m["id"] for m in markers}:
            marker_photo_counts[marker_id] += 1
        if ppm:
            measurements = ", ".join(
                f"ID {m['id']} {m['px_per_mm']:.2f} px/mm" for m in markers if m["px_per_mm"] is not None)
            note = (f"  Photo {i}: {len(markers)} layout marker(s), median {ppm:.2f} px/mm "
                    f"({measurements}); {view_note}")
            if scale_warning:
                note += f"; {scale_warning}"
            notes.append(note)
        elif markers:
            notes.append(f"  Photo {i}: marker IDs {', '.join(str(m['id']) for m in markers)} visible; {view_note}")
        else:
            notes.append(f"  Photo {i}: no layout markers visible; {view_note}")
    photos_with_markers = sum("layout marker(s)" in n or "marker IDs" in n for n in notes)
    print(f"   {len(photos)} keyframes, layout markers found in {photos_with_markers}")
    print("   marker coverage: " + ", ".join(
        f"ID {marker_id} {MARKER_LAYOUT[marker_id]}: {count} photo(s)"
        for marker_id, count in marker_photo_counts.items()))
    side_markers = {"front": (0,), "right": (1, 4), "back": (2, 4, 5), "left": (3, 5)}
    missing_sides = [side for side, marker_ids in side_markers.items()
                     if not any(marker_photo_counts[marker_id] for marker_id in marker_ids)]
    if missing_sides:
        print("   WARNING: no photos cover the " + ", ".join(missing_sides)
              + " side(s); those parts of the object may be poorly reconstructed.")
    if args.marker_mm and photos_with_markers == 0:
        print("   WARNING: marker not found in any frame, so dimensions will be guesses.")

    # 3. first model
    print(f"2. Generating the model with {args.model} (effort={args.effort})...")
    prompt = FIRST_PROMPT.format(
        marker=args.marker_mm or "UNKNOWN",
        scale_notes="\n".join(notes) if args.marker_mm else "  (no marker used; estimate size from context)",
        describe=f"\nHint from the user: {args.describe}\n" if args.describe else "")
    reply = ask(client, args.model, args.effort, prompt, photos, log, "initial")
    model_name = extract_name(reply)
    code = extract_code(reply)
    if not code:
        sys.exit(f"The model didn't return a code block. See {log}")

    # 4-5. run, fix, render, refine
    best = None
    for rnd in range(args.rounds + 1):
        work = out / f"round{rnd}"
        print(f"3.{rnd} Building round {rnd} in FreeCAD...")
        code, info = build_until_it_runs(client, args, fc, code, work, log)
        if not info:
            print("   Could not get a working script this round.")
            break
        best = (work, code, info)
        export_colored_models(work)
        rend = render(work / "model.stl", work / "render.png", info)
        print(f"      bounding box {info['xlen']:.1f} x {info['ylen']:.1f} x {info['zlen']:.1f} mm")
        if rnd == args.rounds:
            break
        print("      comparing the render against the photos...")
        warn = "" if info["valid"] and info["solids"] == 1 else \
            f"WARNING: the result is {'not a valid solid' if not info['valid'] else ''} with {info['solids']} solids; it should be one valid solid."
        reply = ask(client, args.model, args.effort,
                    REFINE_PROMPT.format(code=code, warn=warn,
                                         scale_hint=f"The marker is {args.marker_mm} mm wide." if args.marker_mm else ""),
                    photos + [("Render of your current model:", rend)], log, f"refine {rnd + 1}")
        if reply.strip().upper().startswith("MATCH"):
            print("      model says it matches.")
            break
        code = extract_code(reply) or code

    if not best:
        sys.exit(f"No working model produced. See {log} for the model's replies and errors.")
    finish(out, *best, args, model_name)


if __name__ == "__main__":
    main()
