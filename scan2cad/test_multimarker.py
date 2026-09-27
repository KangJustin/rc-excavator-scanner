from pathlib import Path

import cv2
import numpy as np

import scan2cad as s


OUT = Path("synthetic_markers")
OUT.mkdir(exist_ok=True)
INPUTS = OUT / "inputs"
ANNOTATED = OUT / "annotated"
INPUTS.mkdir(exist_ok=True)
ANNOTATED.mkdir(exist_ok=True)
ARUCO = cv2.aruco.getPredefinedDictionary(s.ARUCO_DICT)


def background():
    image = np.full((600, 900, 3), 150, dtype=np.uint8)
    for x in range(0, image.shape[1], 100):
        cv2.line(image, (x, 0), (x, image.shape[0] - 1), (145, 145, 145), 1)
    return image


def place_marker(image, marker_id, size, x, y):
    margin = max(size // 8, 8)
    marker = cv2.aruco.generateImageMarker(ARUCO, marker_id, size)
    patch = np.full((size + 2 * margin, size + 2 * margin), 255, dtype=np.uint8)
    patch[margin:margin + size, margin:margin + size] = marker
    image[y:y + patch.shape[0], x:x + patch.shape[1]] = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)


images = []

multi = background()
place_marker(multi, 0, 120, 80, 350)
place_marker(multi, 1, 115, 650, 340)
place_marker(multi, 4, 118, 390, 100)
images.append(("01_multi_consistent", multi))

disagree = background()
place_marker(disagree, 2, 150, 100, 330)
place_marker(disagree, 3, 80, 680, 120)
images.append(("02_multi_disagree", disagree))

single = background()
place_marker(single, 5, 110, 390, 350)
images.append(("03_single", single))

none = background()
cv2.putText(none, "NO MARKERS", (280, 310), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (80, 80, 80), 3)
images.append(("04_none", none))

for name, image in images:
    cv2.imwrite(str(INPUTS / f"{name}.png"), image)

loaded = s.load_frames(INPUTS, video_step=1)
frames = s.pick_keyframes(loaded, k=10)
print(f"Selected {len(frames)} of {len(loaded)} synthetic keyframes: {', '.join(name for name, _ in frames)}")
print("Scale/view notes for every synthetic input, including any frame rejected for blur:")
for name, image in loaded:
    annotated, ppm, markers, view_note, warning = s.detect_marker(image, marker_mm=60)
    cv2.imwrite(str(ANNOTATED / f"{name}.png"), annotated)
    measurements = ", ".join(
        f"ID {m['id']}={m['px_per_mm']:.2f} px/mm" for m in markers
    ) or "none"
    print(f"{name}: median={ppm if ppm is not None else 'none'}; {measurements}")
    print(f"  {view_note}")
    if warning:
        print(f"  {warning}")
