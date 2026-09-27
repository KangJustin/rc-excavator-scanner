# RC Excavator Object Scanner: capture & reconstruction

Independent steps:

1. **`capture/`**: take photos from the C270 webcam while someone drives the excavator around the object.
2. **`reconstruct/`**: turn a folder of photos into a 3D mesh (`.ply` + `.stl`) plus a preview image.
3. **`measure/`**: read that mesh and extract dimensions, volume and shape.
4. **`cad/`**: turn that mesh into a CAD model in FreeCAD, automatically.

```
capture/        webcam preview + hotkeys, numbered photos, CSV log
reconstruct/    COLMAP sparse → dense (COLMAP or OpenMVS) → Open3D Poisson mesh → preview.png
measure/        mesh.stl → real-world dimensions, volume, shape fits → measurements.json (for CAD / AI)
cad/            mesh.stl → parametric FreeCAD model (sphere / cylinder / box) or exact solid; STEP + STL + fidelity report
tests/          157 tests; need no camera, COLMAP or OpenMVS (they use fakes)
```

## Quick start: a full run (photos -> CAD model), e.g. a Coke can

Type these exactly. **`latest` means "the newest session"**, so there is no folder name to fill in (earlier docs used a
`<session>` placeholder that got typed literally: don't).

```bash
cd ~/rc-excavator-scanner && source .venv/bin/activate

python -m reconstruct --doctor                     # 1. install check: expect READY
python -m capture --list-cameras                   # 2. which index is the C270? (look at camera_snapshots/*.jpg)
python -m capture --camera-index 1 --check         #    crisp? bright enough? then close it
python -m capture --camera-index 1                 # 3. SHOOT: SPACE = photo, N = new session, Q = quit
python -m reconstruct latest                       # 4. photos -> mesh + preview.png
open reconstructions/*/preview.png                 #    look at it before going on (newest is the one you want)
python -m cad latest --known-diameter-mm 66 --known-height-mm 122 --use-known-sizes   # 5. mesh -> parametric FreeCAD model
```

Then look at the results in `reconstructions/<newest>/cad/`: `*_appearance.png` (scan vs model), `*.FCStd` (open in FreeCAD),
`*.step`, `*.stl`.

**Scale.** Photogrammetry has no scale, so give one real size. For a standard 12 oz can that's about **66 mm diameter x 122 mm
tall** (measure yours). The tools scale from the fitted diameter when the shape fit is good; when it is poor (a lumpy scan) they
scale from the **standing height above the table** instead, and check the width. Either way the second size is an
**independent check**, printed as e.g. `independent check, width: measured 61.3 mm vs 66 mm expected (-7.2%)`.

**Objects that will not scan.** Mirror-polished metal (a steel bowl), glass and plain glossy plastic have no stable texture of their own:
they show reflections that change with every camera position, so photogrammetry fails on them and leaves a **hole** in the mesh. If the
whole object fails, the tools now say `No object found: the reconstruction contains only a flat surface`, and if the sizes you give
disagree with the scan by more than 25% they stop (`--allow-mismatch` overrides). Fixes: use a matte/textured object, or dull the
shine (a thin coat of removable matte spray or talc/dry shampoo, masking tape, or a printed sticker pattern).

**Ideal vs scanned.** `--use-known-sizes` builds a clean 66 x 122 mm cylinder from the sizes you gave (not the scan's lumpy fit) and
reports how far the scan strays from it. If you give both a diameter and a height and the scan is too lumpy to classify, a cylinder is
assumed (and it says so). Without the flag you get the scan's own fit.

**Colour.** It comes from the photos, so *dark photos give a dark model* (a real run: the can measured about RGB (50, 0, 0) in the raw
frames). `reconstruct` warns when photos are dark (median brightness under 90/255) and `cad` warns when the colour is very dark;
add light and reshoot, or set it yourself: `--color "#c8102e"`.

**Shooting**: 30-60 photos, a full 360 degrees, same distance, pause before each shot (watch the sharpness readout), a few from higher up.
Board flat on the table beside the object (not under it, not covered); **bright, diffuse light from several sides**, no bare-lamp glare on the can.

## Excavator-mounted camera test: checklist

**Mount and cable**
- Camera where the bucket was, looking along the arm. Fixed-focus C270: at your working distance, `python -m capture --camera-index N --check`
  and open `check_frame.jpg`: the object must be **crisp**, not soft. (Too close is the usual cause; back off.)
- Tape the USB cable so it can't snag, and leave slack for a full circle. Keep the laptop where the cable reaches all the way round.
- **Light before anything else.** The preview shows `light NN/255`: it must be green (90+). Last run's frames were dark (can ~RGB (50,0,0)).

**Path**
- Turn the turret so the arm points at the object, then drive the tracks around it: the camera keeps facing the centre.
  Stick masking-tape marks on the floor every ~15-20 degrees on a circle of about 40-60 cm radius so the driver can stop at each mark.
- **Two rings** beat one: a low ring, then raise the boom and do a higher ring (the top of the object is missing otherwise, and
  `reconstruct` warns when all shots are at the same height).
- Scale board flat on the floor beside the object, off its path, in view from many places. Object **not** touching it.

**Shooting (two people)**
- Driver moves to a mark and **stops**. Wait ~1 s for the vibration to die, then the other person presses SPACE (sharpness readout green).
- 40-60 photos total, ~15-20 degrees apart, full 360 degrees, same distance each time.
- If the excavator drifts off the circle, that's fine as long as the object stays in frame at a similar size.

**Then**: `python -m reconstruct latest` then `python -m cad latest --known-diameter-mm 66 --known-height-mm 122 --use-known-sizes`.

## 📸 Shooting tips (for whoever is driving)

- **20-30 photos.** Fewer than ~15 usually fails; more than ~40 just slows things down.
- **Go a full 360°.** Steps of about 12-18° between photos: every photo must overlap the last by most of the frame.
- **Keep the object centred and the same size in frame**, i.e. the same rough distance the whole way round.
- **Pause before every shot.** Stop moving, count "one, two", then press SPACE. Motion blur is the #1 killer. The on-screen `sharpness` readout turns red and says "SOFT" when the frame is blurry: wait and press again.
- **At least a couple of shots from a different height/angle** (higher up, looking down at the object) even if it means tilting the arm. Without them the top of the object will be missing. The reconstruction warns if every shot is at the same height.
- **Give it something to look at.** Shiny, plain, transparent or repeating-pattern objects (soda cans!) are the hardest. Use a matte, busy, non-repeating surface (newspaper, a rug), avoid glare and hard reflections, use diffuse light. Matte tape or marker dots on the object help a lot.
- **Don't move the object** or let it be moved (or people walking through frame) during the orbit.
- The C270 is fixed-focus: if photos look soft even when the excavator is still, back off a bit. Check with `python -m capture --check`.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # numpy, opencv-python, pyserial, open3d, pytest
```

`pyserial` is only used by the unrelated `control/` code (see the last section); capture and reconstruct don't need it.

### COLMAP (required for step 2)

| OS | Install |
|---|---|
| **macOS** | `brew install colmap` (Homebrew currently ships 4.2.0). **No CUDA on macOS**, so COLMAP's own dense stage can't run: use OpenMVS below. |
| **Linux (Debian/Ubuntu)** | `sudo apt install colmap`. Per COLMAP's docs, packages from the default repositories are **built without CUDA**. For GPU dense reconstruction, build from source with CUDA: <https://colmap.github.io/install.html> |

Check what you have: `colmap help` prints a header, and for most builds it says `with CUDA` or `without CUDA`.
If `colmap` isn't on your `PATH`, set `COLMAP_BIN` in `reconstruct/config.py` (or `export COLMAP_BIN=/path/to/colmap`, or pass `--colmap-bin`).

### OpenMVS (CPU alternative for the dense stage)

Use this if you have no NVIDIA GPU (any Mac; most laptops). It is slower than COLMAP-on-GPU but works everywhere.
**There is no Homebrew formula for OpenMVS** (checked with `brew info openmvs`), so build it from source. From the
[official build instructions](https://github.com/cdcseacave/openMVS/wiki/Building):

```bash
# Linux (Ubuntu) build dependencies, from the OpenMVS wiki:
sudo apt install git cmake autoconf autoconf-archive automake libtool bison gfortran pkg-config \
  libxi-dev libx11-dev libxft-dev libxtst-dev libxext-dev libxrandr-dev libxinerama-dev libxcursor-dev \
  xorg-dev libgl-dev libglu1-mesa-dev nasm libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libswresample-dev
# It also needs Eigen 3.4+, OpenCV, Ceres, CGAL 4.2+, Boost 1.56+, and a C++17 compiler (GCC 9+): install
# those from your package manager if cmake complains (libeigen3-dev libopencv-dev libceres-dev libcgal-dev libboost-all-dev).

# macOS: I have NOT built this on macOS. A reasonable starting point is
#   brew install cmake boost eigen opencv cgal ceres-solver
# then follow the wiki if cmake reports anything missing.

git clone --recurse-submodules https://github.com/cdcseacave/openMVS.git
cd openMVS && mkdir make && cd make
cmake ..                      # CUDA is optional; without it OpenMVS runs on CPU
cmake --build . -j4
cmake --install .
```

Then tell the pipeline where the binaries are (it needs `InterfaceCOLMAP` and `DensifyPointCloud`). If they aren't on
`PATH`, find them with `find / -name DensifyPointCloud -type f 2>/dev/null` and set `OPENMVS_BIN_DIR` in
`reconstruct/config.py` (or `export OPENMVS_BIN_DIR=...`, or pass `--openmvs-bin-dir`).

### Check your install: `--doctor`

```bash
python -m reconstruct --doctor
```

Run this after installing COLMAP and building OpenMVS, before a real capture. It checks, without touching any photos:

- COLMAP is found (on `PATH` or at `COLMAP_BIN`), actually runs, and whether its header says **with CUDA** or **without CUDA**
- OpenMVS's `InterfaceCOLMAP` and `DensifyPointCloud` are found (in `OPENMVS_BIN_DIR` or on `PATH`) **and launch**
  (catches a missing shared library). If they aren't where you pointed, it looks for them nearby and tells you the folder to use.
- Open3D (and OpenCV) import
- **which dense backend `auto` would really use**, and why

It ends with `READY` or `NOT READY` (exit code 0 / 1). Paste its output when asking for help: it includes platform, versions and paths.

### Which dense backend?

| You have | Use |
|---|---|
| NVIDIA GPU + CUDA-enabled COLMAP (some other machine) | `auto` picks it, or force with `--dense-backend colmap` (fast) |
| No NVIDIA GPU (any Mac, incl. all Apple Silicon: no CUDA, ever) | `auto` falls back to OpenMVS on its own, or force with `--dense-backend openmvs` (CPU, slower) |
| Not sure yet | leave the default, `auto`: COLMAP if it can, else OpenMVS |

The default lives in `reconstruct/config.py` (`DENSE_BACKEND`). Both backends produce the same `dense/fused.ply`, so
everything after them is identical. Adding a third backend means subclassing `DenseBackend` in `reconstruct/dense.py`.

## Step 1: capture

```bash
python -m capture               # live preview
python -m capture --check       # sanity-check the feed first (resolution, fps, sharpness, exposure)
python -m capture --list-cameras
```

| Key | Action |
|---|---|
| **SPACE** | take a photo |
| **N** | start a new session folder (ignored if the current one is still empty) |
| **Q** / ESC | quit |

The window shows a big **SHOTS: n** counter (white below 20, green for 20-30, amber above), the session name, a live
sharpness readout, and a confirmation message after each shot.

Photos go to `captures/<YYYY-MM-DD_HHMM>/`, created on the first photo (so no empty folders):

```
captures/2026-09-25_2130/
    0001.jpg 0002.jpg ...
    captures.csv        index, file, timestamp (ISO), unix_time, width, height, sharpness, extra
```

**Picking the camera** (the MacBook's built-in camera and a USB webcam get different indices):

```bash
python -m capture --list-cameras     # saves camera_snapshots/camera_<N>.jpg for each camera found
python -m capture --camera-index 1   # aliases: --camera 1, --device 1
export CAMERA_INDEX=1                # optional: make a choice stick (add to ~/.zshrc)
```

OpenCV can't report device *names*, so `--list-cameras` saves a snapshot from each one: open them to see which is
which (the C270 will show whatever you've pointed it at). The preview shows `camera N WxH` in its bottom-right corner
so you can confirm you're on the right one. Indices can shift when you plug/unplug cameras or when an iPhone shows
up as a camera (Continuity Camera). Note the snapshots include a picture from the built-in camera; they're
git-ignored, delete the folder whenever you like.

Two sessions started in the same minute get `_2`, `_3` suffixes. Deleting a bad shot is safe: numbering continues past
the highest number, so nothing is overwritten. On macOS, allow camera access for your terminal
(System Settings → Privacy & Security → Camera). Options: `--camera N`, `--session-root`, `--lock-exposure`
(best effort; consistent exposure helps matching).

## Step 2: reconstruct

```bash
python -m reconstruct captures/2026-09-25_2130
python -m reconstruct captures/2026-09-25_2130 --dense-backend openmvs
```

Output goes to `reconstructions/<session name>/` (deliberately *outside* the photo folder: COLMAP scans photo folders
recursively and would ingest its own output):

| File | What |
|---|---|
| `preview.png` | 6 views of the mesh (front/right/back/left, top, underside) to sanity-check without a viewer |
| `mesh.stl` / `mesh.ply` | the result (`.ply` keeps colours) |
| `sparse_points.ply` | sparse cloud: open in MeshLab/CloudCompare to check the camera solve |
| `dense/fused.ply`, `cleaned_points.ply` | dense cloud, and the filtered cloud that was meshed |
| `scene.json`, `report.json`, `logs/*.log` | camera-path analysis, summary, full COLMAP/OpenMVS output per step |

The mesh is **rotated Z-up** (using the plane the cameras orbited in) and sits on z=0, so it comes out the right way
up (`--no-align` to keep COLMAP's arbitrary frame). **Scale is arbitrary** (structure-from-motion can't know real
size): the extents are printed in model units; measure the real object and pass `--scale`.

Useful options: `--only sparse|dense|mesh|preview` (run one stage against an existing workspace),
`--remove-plane` (delete the table), `--largest-cluster`, `--crop`, `--poisson-depth`, `--allow-partial`, `--no-preview`,
`-v` (echo COLMAP output). `python -m reconstruct --help` lists everything.

### It won't look frozen

COLMAP and OpenMVS report no real progress, and dense reconstruction can run 30+ minutes with no output. So while any
external command runs, a background thread prints a liveness line every 10 seconds (change `HEARTBEAT_SECONDS` in
`reconstruct/config.py`; `0` turns it off):

```
[openmvs_densify] running... 04:32 elapsed
[openmvs_densify] done in 18:47          (or "failed after 04:32" if the tool exited non-zero)
```

It shows only the stage name and elapsed time (no fake percentage). It never touches the tool or its exit code, and it
isn't written to `logs/*.log`, which contain only the tool's own output. Stage names: `feature_extractor`,
`exhaustive_matcher`, `mapper`, `model_converter`, `image_undistorter`, `patch_match_stereo`, `stereo_fusion`,
`openmvs_interface`, `openmvs_densify`.

### Errors it diagnoses, and what it tells you to do

| Problem | How it's detected | Message |
|---|---|---|
| **Too few matched features** (not enough overlap) | graph of geometrically verified photo pairs | smaller steps (≤15-20°), constant object size, no blur; names disconnected groups and isolated photos |
| Too few features per photo | keypoints in COLMAP's database | textureless/shiny object advice |
| **Degenerate: straight-line camera path** | camera centres of the solved model are near-collinear | go *around* the object, not past it |
| **Degenerate: not enough angular coverage** | angular span of solved cameras around the object (< 120°; warns below 300°) | go the full 360°; names the two photos either side of the biggest gap. `--allow-partial` continues anyway |
| All shots at the same height | spread of camera elevation (warning) | add a few from higher up |
| Mapper found no model / few photos registered | mapper output | more parallax/overlap; lists unregistered photos |
| **COLMAP not found** | executable lookup | how to install / set `COLMAP_BIN` |
| **OpenMVS not found** | executable lookup | how to build / set `OPENMVS_BIN_DIR` |
| COLMAP dense without CUDA | `colmap help` header, and the stereo step's own error | suggests OpenMVS (auto mode switches to it by itself) |
| Too few / blurry / mixed-size photos | preflight | fix or delete them |

Every error prints a "What to try" line and the log file to read.

### Splitting stages across machines

The workspace folder is the hand-off, e.g. sparse on a laptop, dense on a GPU box:

```bash
python -m reconstruct captures/A --only sparse
# copy captures/A and reconstructions/A to the other machine, then:
python -m reconstruct captures/A --only dense      # auto picks COLMAP on a CUDA box, OpenMVS elsewhere
python -m reconstruct captures/A --only mesh        # needs only Open3D
```

### Known issue: Open3D Poisson crashes

Open3D 0.20's multithreaded Poisson aborted the whole process about 1 run in 5 on macOS arm64 (no Python exception).
The pipeline defaults to `--poisson-threads 1`, which was stable across 40 consecutive runs. Raise it only if you've
verified your machine.

## Step 3: measure (`python -m measure`)

Turns a reconstructed mesh into numbers you can give to an AI (or a person) to build a parametric model.

```bash
python -m measure reconstructions/2026-09-25_2130 --scale-mm 65.4
```

It reads `reconstructions/<session>/mesh.stl`, prints a summary block, and saves `measurements.json` next to it
(the JSON also contains the text block as `summary_text`, so either can be pasted into a prompt).

**You must give it a scale.** Photogrammetry has arbitrary scale, so the mesh is in unknown "model units".
`--scale-mm` is **millimetres per model unit**. Measure one real dimension of the object with a ruler, note the same
dimension in the un-scaled output, and divide: a can that's really 122 mm tall but reports `1.87` needs `--scale-mm 65.2`.
Without it the output says **NOT SCALED** (and the JSON has `"provided": false`) so nobody mistakes the numbers for
millimetres. If you already scaled the mesh with `reconstruct --scale`, don't scale twice: pass `--scale-mm 1`.

What you get, all in mm:

| Item | Details |
|---|---|
| **Bounding boxes** | axis-aligned X/Y/Z (Z is up when the mesh was aligned by reconstruct), and the tightest oriented box as length/width/height sorted longest to shortest. **"height" there is the shortest side, not necessarily vertical**; `height_above_base_plane_mm` is the true standing height |
| **Volume** | mm³ and mL, plus the convex-hull volume as an upper bound. If the mesh is open and there is no base plane, volume is reported as unreliable rather than guessed |
| **Surface area** | exposed surface only (excludes the face touching the table) |
| **Sphere and cylinder fits** | fitted diameter / height / axis, the RMS distance of the surface from the fitted shape, and a verdict: `cylinder`, `sphere`, or `neither`, with fit quality good (< 4% of radius), rough (< 10%) or poor |
| **Box fill ratio** | volume ÷ oriented-box volume (box ≈ 1.0, cylinder ≈ 0.79, sphere ≈ 0.52): a cheap hint when both fits are poor |
| **Warnings** | missing scale, skipped plane removal, open mesh, leftover pieces, rough fits |

**Isolating the object from the table.** RANSAC (Open3D `segment_plane` on the mesh vertices) finds the dominant plane,
but "biggest plane" isn't always the table (on a box it can be the box's own top face). So a plane is used only if it
is near-horizontal (when the mesh is Z-up), holds ≥ 10% of the vertices, has a connected object resting on it, and is
clearly wider than that object. Otherwise the next plane is tried, and after three misses it **skips plane removal
and measures the whole mesh**, saying why. Once a plane is accepted the object is assumed to rest on it, so its
unseen underside is taken to be on the plane (this is what makes the height and volume of, say, a ball right).
`--no-plane-removal` measures everything; `--plane-tol` sets the plane thickness (fraction of the mesh diagonal).

Checked on synthetic scenes with known answers (a 66 × 122 mm can, an 80 mm ball and a box, each standing on a noisy
table and meshed by the real pipeline): dimensions within ~1%, volumes within 0.1-1.5%, and the can/ball/box
correctly told apart. Real photogrammetry meshes are noisier and rougher than that, so treat the numbers as good
estimates, and sanity-check them against a ruler before sending them to CAD.

### From measurements to a FreeCAD model

`freecad/scanned_sphere.FCMacro` builds an editable parametric model in FreeCAD (open it in FreeCAD's macro editor and
Execute; tested on FreeCAD 1.1.3). It was written from the measurements of the basketball scan
(`reconstructions/2026-09-25_2257`: sphere fit *good*, 3.2% of radius; box-fill 0.56, which is what a sphere gives).

```
Params (Spreadsheet)     Diameter / Radius / VolumeRef        <- edit here, the model regenerates
Body
  ProfileSketch          fully constrained half-disc; its "Diameter" constraint is bound to Params.Diameter
  SphereRevolution       360-degree revolve => sphere sitting on Z = 0
```

`DIAMETER_MM` is set to the ball's measured **44 mm**. The scan agrees: scaled by 44 mm (79.594 mm per model unit, from the sphere
fit), the independent measurements came out at height-above-floor 44.3 mm, oriented box 44.3-44.8 mm and axis-aligned
box 44.9-45.4 mm, i.e. within ~1-3%. (If you copy the macro for a new object and haven't measured it, set
`SCALE_CONFIRMED = False`: it warns and tags the document "PLACEHOLDER SIZE".) Optional `EXPORT_STL_PATH` writes a
print-ready STL and `SAVE_FCSTD_PATH` saves the editable document; the results are checked in as
`freecad/ball_44mm.FCStd` (open in FreeCAD; edit `Params.Diameter` to resize) and `freecad/ball_44mm.stl`.

**Volume caveat:** for the scan, `measure` reports the measured volume (49.7 mL) and the fitted-sphere volume (44.6 mL). They
differ because the ball's underside is never photographed, which makes the measured one over-count. When the shape fit is
*good*, use the fitted-shape volume (`primitive_fit.<shape>.volume_mm3`); `measure` now prints it and warns when they disagree. To write a similar macro for a different scan, give an AI the
`measurements.json` (or its `summary_text`) and this macro as the pattern: cylinder scans want a Pad, not a Revolution.

## Scale board (for automatic real-world size)

`scale_board/` holds a printable **ChArUco board** whose true size is known exactly, so software can later turn "model
units" into millimetres with no ruler. (The detector/scale step that uses it isn't built yet: this is the board.)

| File | What |
|---|---|
| `charuco_letter.pdf`, `charuco_a4.pdf` | the board: 5 × 7 squares of **30.0 mm** (150 × 210 mm), 22.5 mm ArUco tags, `DICT_4X4_50`; a 100 mm check ruler and instructions are on the sheet |
| `charuco_board.json` | the same numbers as data (what the software will read) |
| `charuco_preview.png` | quick look |

**To print:** open the PDF and print at **100% / "Actual size"**, never "fit to page" (fit-to-page silently rescales it and
ruins the scale). Then lay a real ruler on the printed 100 mm line: it must read 100 mm. Also measure one square; if it isn't
30.0 mm, put the true value in `printed_square_mm_measured` in the JSON. Tape the sheet flat, on the same surface the
object rests on, and keep it in view of the camera.

It is generated by `python -m measure.board` from one dictionary (`BOARD` in `measure/board.py`), so the PDF and the JSON
can't disagree; a test fails if you edit one without regenerating. The PDF is drawn as vector rectangles (7 KB, no
resampling). I rendered both PDFs at 300 DPI and checked that all 24 corners and 17 tags are detected and the squares
measure 29.99-30.00 mm; the remaining uncertainty is your printer, which is what the ruler check is for.

## Step 4: CAD (`python -m cad`)

Takes the mesh and produces a CAD model, with no hand-written macro:

```bash
python -m cad reconstructions/2026-09-25_2257          # scale comes from measurements.json (or pass --scale-mm)
python -m cad reconstructions/<session> --shape solid  # force an exact, non-parametric copy
```

It isolates the object from the table, decides what it is, builds it in headless FreeCAD, and **measures how far the model is
from the scan**. Output goes to `<reconstruction>/cad/`: `<name>.FCStd` (editable), `.step`, `.stl`, `.FCMacro` (the script that
rebuilds it, editable), `_preview.png`, and `cad_report.json`.

| Detected | What you get |
|---|---|
| **sphere** | PartDesign Body: `Params` spreadsheet → half-disc sketch (dimension bound to `Params.Diameter`) → Revolution |
| **cylinder** | `Params` (Diameter, Height) → circle sketch → Pad |
| **box** | `Params` (Length, Width, Height) → fully constrained rectangle sketch → Pad. Height is the vertical side |
| **none of these** | **NOT parametric.** The scan's own surface, closed and converted to a valid solid (exact, including noise/bumps). You can print it and use it in assemblies, but you can't type a new dimension |

Edit a parametric model by changing cells in `Params` and pressing Ctrl+R. **Fidelity** is the distance from the scan's surface to the
model (rms / p95 / max in mm, and as a % of the model's size): under 2% is graded *excellent*, under 5% *good*.

**Colour and appearance.** The model is given the object's colour, taken from the scan's vertex colours (`mesh.ply`): highlights and
shadows are trimmed, colours are clustered by hue (not brightness, or one orange ball splits into "dark, mid and light orange"),
and the model gets the biggest hue if it covers at least half the object, otherwise the average. You also get
`<name>_appearance.png` (the scan in its true photographed colours above the CAD model) and `<name>_scan_colored.ply` (the scan object
as a coloured 3D file: this is the "what it actually looks like" version, seams and all). Colours are *as seen in the photos*
(the lighting is baked in), not paint values, and the CAD model is a single flat colour: a red-and-silver can becomes red.
FreeCAD stores colour in the `.FCStd` as `[alpha, B, G, R]`; I first wrote R, G, B and an orange ball came out blue, which is how
the byte order was confirmed in the real GUI.

**Honest limits.** This recognises simple shapes; it does not reverse-engineer arbitrary parts into feature trees (a bracket with
holes, fillets and ribs would become the exact-solid fallback, not a parametric model). It doesn't recover edge fillets or fine
detail (a basketball's seams). The scale must be right (`--scale-mm`): the tool refuses to run without one unless you pass
`--unscaled`. The `.FCStd` files are built in headless FreeCAD, which can't store the 3D view state; the tool adds it afterwards
(`cad/guistate.py`: the Body and its tip feature shown, the sketch and construction planes hidden, an isometric camera fitted to
the model), so a model opens showing itself. Checked by opening one in the FreeCAD GUI. (Without that step FreeCAD opens
a PartDesign Body with its tip feature hidden, i.e. an empty view.)

Checked on synthetic scenes with known answers plus your real ball (sphere, 44.0 mm, 0.72 mm rms from the scan, 0.9% of size):
can → cylinder, box → box (both ~0.1-0.3% of size), and a block-with-tower (no primitive fits) → a valid solid within 0.07% of
the true volume. Two bugs found and fixed on the way: hole-filling that left flipped triangles (wrong volume; now checked with
our own consistency test), and the table cut leaving the base floating ~7 mm above the plane (now extended down to it).

## What has and hasn't been verified

- **Tested (157 tests):** photo numbering/CSV/hotkey logic; `--doctor`; the reconstruction orchestration, every error branch, the
  degenerate-path geometry, Z-up alignment, the preview renderer (also inspected by eye), and the meshing, using
  synthetic data and **fake `colmap`/`InterfaceCOLMAP`/`DensifyPointCloud` executables** (`tests/fake_*.py`).
- **Run for real:** the full capture-to-mesh pipeline on 66 photos of a basketball (64 registered, OpenMVS dense, mesh, preview), and `measure` on that mesh (removed the floor, called it a good sphere fit). The FreeCAD macro is run in a real headless FreeCAD 1.1.3 by the tests.
- **Not yet verified:** real COLMAP or OpenMVS on real photos (neither is installed on the
  machine this was written on). The command lines follow COLMAP's current CLI docs and OpenMVS's source, but real
  runs are the true test. **Do a run on real photos before relying on it.**

## Unrelated leftovers

`control/`, `firmware/` and `orbit_capture.py` come from an earlier iteration (automated driving via an ESP32) and are
not part of this pipeline. They are untouched and unused here.
